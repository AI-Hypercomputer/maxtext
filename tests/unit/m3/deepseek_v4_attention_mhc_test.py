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

"""Parity of the m3 DeepSeek-V4 attention and hyper-connection layers against the Hugging Face reference golden.

Every forward output and every parameter and input gradient is compared with the committed float32 golden
(`testdata/deepseek_v4/mini_golden.npz`, produced from the unmodified `transformers` DeepSeek-V4 modules). Parameters,
inputs and cotangents are rebuilt from the golden manifest's hashed-tensor recipe, so the test needs no PyTorch.
"""

from collections.abc import Callable, Mapping
import dataclasses
import functools
import importlib
import io
import json
import pathlib
import re
from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.models.deepseek_v4 import modeling_deepseek_v4 as dsv4
from tests.unit.m3.testing import layer_parity

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "deepseek_v4"
# Approved tolerance rows T1-T8 (float32) and T14 (float64 algebraic equivalence).
_TOL = layer_parity.Tolerance(atol=1e-5, rtol=1e-5, frob_rel=1e-5)
_FP64_TOL = layer_parity.Tolerance(atol=1e-10, rtol=1e-10, frob_rel=1e-10)
_NEG_SCALE = 1e-2


def _load_golden() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  manifest = json.loads((_TESTDATA / "golden_manifest.json").read_text(encoding="utf-8"))
  with np.load(io.BytesIO((_TESTDATA / "mini_golden.npz").read_bytes()), allow_pickle=False) as data:
    arrays = {k: data[k] for k in data.files}
  return manifest, arrays


MANIFEST, GOLDEN = _load_golden()


def _config(**overrides) -> dsv4.DeepseekV4Config:
  return dsv4.DeepseekV4Config.from_hf_config(MANIFEST["hf_config"], dtype="float32", weight_dtype="float32", **overrides)


def _f64_config(**overrides) -> dsv4.DeepseekV4Config:
  return dsv4.DeepseekV4Config.from_hf_config(
      MANIFEST["f64_hf_config"], dtype="float64", weight_dtype="float64", **overrides
  )


def _hashed(key: str, spec: Mapping[str, Any]) -> np.ndarray:
  dtype = np.float64 if spec.get("dtype") == "float64" else np.float32
  return layer_parity.hashed_normal(key, spec["shape"], mean=spec["mean"], std=spec["std"], dtype=dtype)


def _jax_path(hf_name: str) -> tuple[str, str]:
  """Maps a reference parameter name to (m3 parameter path, layout transform).

  Transforms: "I" identity, "T" transpose ([out, in] -> [in, out]), "G" grouped projection
  ([groups * out, in] -> [groups, in, out]).
  """
  name = hf_name.replace("compressor.indexer.scorer.", "compressor.indexer.")
  prefix, rest = "", name
  if name.startswith("compressor.indexer."):
    rest = name.removeprefix("compressor.indexer.")
    prefix = "compressor/indexer/"
    if rest.split(".")[0] in ("kv_proj", "gate_proj", "position_bias", "kv_norm"):
      prefix += "compressor/"
  elif name.startswith("compressor."):
    prefix, rest = "compressor/pool/", name.removeprefix("compressor.")
  if rest == "o_a_proj.weight":
    return prefix + "o_a_proj/weight", "G"
  if rest in ("fn", "hc_fn"):
    return rest, "T"
  if rest.endswith("norm.weight") or rest in ("weight", "sinks", "position_bias", "base", "scale", "hc_base", "hc_scale"):
    return prefix + rest.replace(".", "/"), "I"
  if rest.endswith(".weight"):
    return prefix + rest.removesuffix(".weight").replace(".", "/") + "/kernel", "T"
  raise KeyError(f"No m3 mapping for reference parameter {hf_name!r}")


def _to_jax_layout(arr: np.ndarray, transform: str, groups: int) -> np.ndarray:
  if transform == "I":
    return arr
  if transform == "T":
    return arr.T
  out_total, in_per_group = arr.shape
  return arr.reshape(groups, out_total // groups, in_per_group).transpose(0, 2, 1)


def _set_by_path(tree: dict[str, Any], path: str, value: Any) -> None:
  node = tree
  parts = path.split("/")
  for part in parts[:-1]:
    node = node[part]
  if np.shape(node[parts[-1]]) != np.shape(value):
    raise ValueError(f"Shape mismatch at {path}: module {np.shape(node[parts[-1]])} vs reference {np.shape(value)}")
  node[parts[-1]] = value


class _Component:
  """A golden component bound to an m3 module: reference-loaded parameters, inputs, outputs and gradients."""

  def __init__(self, comp: str, module: nnx.Module | None, cfg: dsv4.DeepseekV4Config, path_fn=_jax_path):
    self.spec = MANIFEST["components"][comp]
    self.comp = comp
    self.module = module
    source = self.spec.get("param_source", comp)
    params_spec = MANIFEST["components"][source]["params"]
    self.transforms = {p["name"]: path_fn(p["name"]) for p in params_spec}
    if module is None:
      self.graphdef, self.state, self.params = None, None, {}
    else:
      self.graphdef, self.state = nnx.split(module)
      pure = nnx.to_pure_dict(self.state)
      for p in params_spec:
        path, transform = self.transforms[p["name"]]
        old = _leaf(pure, path)
        value = _to_jax_layout(_hashed(f"{source}/param/{p['name']}", p), transform, cfg.o_groups)
        _set_by_path(pure, path, jax.device_put(jnp.asarray(value, dtype=old.dtype), old.sharding))
      self.params = pure
      n_module = len(jax.tree.leaves(pure))
      if n_module != len(params_spec):
        raise ValueError(f"{comp}: module has {n_module} parameters, reference has {len(params_spec)}")
    self.groups = cfg.o_groups

  def bind(self, params: Any) -> nnx.Module:
    state = jax.tree.map(lambda x: x, self.state)
    nnx.replace_by_pure_dict(state, params)
    return nnx.merge(self.graphdef, state)

  def inputs(self) -> dict[str, jax.Array]:
    return {i["name"]: jnp.asarray(_hashed(f"{self.comp}/input/{i['name']}", i)) for i in self.spec["inputs"]}

  def reference(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Returns (outputs, cotangents, gradients in m3 layout, gradient map)."""
    np_dtype = np.float64 if self.spec.get("dtype") == "float64" else np.float32
    outs = {k: GOLDEN[f"{self.comp}/out/{k}"] for k in self.spec["outputs"]}
    cots = {
        k: layer_parity.hashed_normal(f"{self.comp}/cotangent/{k}", shape, mean=0.0, std=1.0, dtype=np_dtype)
        for k, shape in self.spec["outputs"].items()
    }
    grads, grad_map = {}, {}
    for name, (path, transform) in self.transforms.items():
      if name in self.spec["zero_grad_params"]:
        grads[name] = np.zeros(np.shape(_leaf(self.params, path)), np_dtype)
      else:
        grads[name] = _to_jax_layout(GOLDEN[f"{self.comp}/grad/param/{name}"], transform, self.groups)
      grad_map[name] = f"params/{path}"
    for i in self.spec["inputs"]:
      grads[f"input:{i['name']}"] = GOLDEN[f"{self.comp}/grad/input/{i['name']}"]
      grad_map[f"input:{i['name']}"] = f"inputs/{i['name']}"
    return outs, cots, grads, grad_map


def _leaf(tree: Any, path: str) -> Any:
  node = tree
  for part in path.split("/"):
    node = node[part]
  return node


def _attention_module(cfg: dsv4.DeepseekV4Config, kind: str, sharding=None) -> nnx.Module:
  layer_idx = MANIFEST["components"][f"attn_{kind}"]["layer_idx"]
  return dsv4.DeepseekV4Attention(cfg, layer_idx, sharding=sharding, rngs=nnx.Rngs(0))


class ParityTestBase(unittest.TestCase):
  """Shared golden parity and negative-control assertions."""

  def assert_parity(
      self,
      comp: _Component,
      fn: Callable[[Any, Any], Mapping[str, jax.Array]],
      tol: layer_parity.Tolerance = _TOL,
  ) -> None:
    """Asserts every output and parameter/input gradient of `fn` matches the golden at `tol`."""
    outs, cots, grads, grad_map = comp.reference()
    report = layer_parity.check_forward_and_grads(
        cand_fn=fn,
        cand_params=comp.params,
        cand_inputs=comp.inputs(),
        ref_outputs=outs,
        cotangents=cots,
        ref_grads=grads,
        grad_map=grad_map,
        fwd_tol=tol,
        grad_tol=tol,
    )
    self.assertTrue(report.passed, "\n".join(report.failures))

  def assert_negative_control(self, comp: _Component, fn, target_path: str) -> None:
    """Asserts a 1% perturbation of `target_path` makes the forward comparison fail."""
    outs, _, _, _ = comp.reference()
    fired, metrics = layer_parity.negative_control_fires(
        cand_fn=fn,
        cand_params=comp.params,
        cand_inputs=comp.inputs(),
        ref_outputs=outs,
        fwd_tol=_TOL,
        target_path=target_path,
        rel_scale=_NEG_SCALE,
        seed=0,
    )
    self.assertTrue(fired, f"Perturbing {target_path} was not detected: {metrics}")


class NormRopeAndProjectionTest(ParityTestBase):
  """Rows T1-T5: RMSNorm, interleaved RoPE, grouped output projection, hyper-connection, and hyper-head."""

  def setUp(self):
    super().setUp()
    self.cfg = _config()

  def test_rmsnorm_t1(self):
    module = dsv4.DeepseekV4RMSNorm(
        self.cfg.hidden_size, eps=self.cfg.rms_norm_eps, param_dtype=jnp.float32, sharding=None, name="norm"
    )
    comp = _Component("rmsnorm", module, self.cfg)

    def fn(p, inp):
      return {"y": comp.bind(p)(inp["x"])}

    self.assert_parity(comp, fn)
    self.assert_negative_control(comp, fn, "weight")

  def test_unweighted_rmsnorm_t1(self):
    comp = _Component("unweighted_rmsnorm", None, self.cfg)
    self.assert_parity(comp, lambda p, inp: {"y": dsv4.unweighted_rms_norm(inp["x"], self.cfg.rms_norm_eps)})

  def test_rope_inv_freq_bit_exact_and_rotation_t2(self):
    for rope_type in ("main", "compress"):
      np.testing.assert_array_equal(
          dsv4.rope_inv_freq(self.cfg, rope_type), GOLDEN[f"rope_tables/out/{rope_type}_inv_freq"], err_msg=rope_type
      )
      start, stop = MANIFEST["components"]["rope_tables"]["positions"]
      cos, sin = dsv4.rope_cos_sin(
          tuple(dsv4.rope_inv_freq(self.cfg, rope_type)), jnp.arange(start, stop)[None], jnp.float32
      )
      for name, value in (("cos", cos), ("sin", sin)):
        ok, metrics = layer_parity.compare(np.asarray(value), GOLDEN[f"rope_tables/out/{rope_type}_{name}"], _TOL)
        self.assertTrue(ok, f"{rope_type} {name}: {metrics}")

    comp = _Component("rope_apply", None, self.cfg)

    def fn(_, inp):
      x = inp["x"]  # [B, H, S, D] as in the reference
      batch, _, seq, _ = x.shape
      pos = jnp.broadcast_to(jnp.arange(seq), (batch, seq))
      x_bshd = jnp.swapaxes(x, 1, 2)
      out = {}
      for rope_type in ("main", "compress"):
        cos, sin = dsv4.rope_cos_sin(tuple(dsv4.rope_inv_freq(self.cfg, rope_type)), pos, x.dtype)
        out[f"{rope_type}_rotated"] = jnp.swapaxes(dsv4.apply_rotary(x_bshd, cos, sin), 1, 2)
        out[f"{rope_type}_inverse"] = jnp.swapaxes(dsv4.apply_rotary(x_bshd, cos, -sin), 1, 2)
      return out

    self.assert_parity(comp, fn)

  def test_grouped_linear_t3(self):
    cfg = self.cfg
    in_g = cfg.num_attention_heads * cfg.head_dim // cfg.o_groups
    module = dsv4.DeepseekV4GroupedLinear(
        cfg, in_g, cfg.o_lora_rank, cfg.o_groups, sharding=None, rngs=nnx.Rngs(0), name="o_a_proj"
    )
    comp = _Component("grouped_linear", module, cfg, path_fn=lambda name: ("weight", "G"))

    def fn(p, inp):
      return {"y": comp.bind(p)(inp["x"])}

    self.assert_parity(comp, fn)
    self.assert_negative_control(comp, fn, "weight")

  def test_hyper_connection_t4(self):
    module = dsv4.DeepseekV4HyperConnection(self.cfg, sharding=None, rngs=nnx.Rngs(0), name="attn_hc")
    comp = _Component("hyper_connection", module, self.cfg)

    def fn(p, inp):
      post, comb, collapsed = comp.bind(p)(inp["x"])
      return {"post": post, "comb": comb, "collapsed": collapsed}

    self.assert_parity(comp, fn)
    self.assert_negative_control(comp, fn, "fn")
    out = fn(comp.params, comp.inputs())
    np.testing.assert_allclose(np.asarray(out["comb"]).sum(axis=-2), 1.0, atol=1e-5)
    post = np.asarray(out["post"])
    self.assertTrue(np.all((post >= 0) & (post <= 2)))

  def test_hyper_head_t5(self):
    module = dsv4.DeepseekV4HyperHead(self.cfg, sharding=None, rngs=nnx.Rngs(0), name="hc_head")
    comp = _Component("hyper_head", module, self.cfg)

    def fn(p, inp):
      return {"y": comp.bind(p)(inp["x"])}

    self.assert_parity(comp, fn)
    self.assert_negative_control(comp, fn, "hc_fn")


class AttentionTest(ParityTestBase):
  """Rows T6-T8: sliding-window, heavily compressed and compressed sparse attention."""

  def setUp(self):
    super().setUp()
    self.cfg = _config()

  def _component(self, kind: str, sharding=None) -> _Component:
    return _Component(f"attn_{kind}", _attention_module(self.cfg, kind, sharding), self.cfg)

  def _check(self, kind: str, negative_paths: tuple[str, ...]) -> _Component:
    comp = self._component(kind)

    def fn(p, inp):
      return {"out": comp.bind(p)(inp["hidden"])[0]}

    self.assert_parity(comp, fn)
    for path in negative_paths:
      self.assert_negative_control(comp, fn, path)
    return comp

  def test_sliding_attention_t8(self):
    self._check("sliding", ("q_b_proj/kernel", "kv_proj/kernel", "o_a_proj/weight", "sinks"))

  def test_heavily_compressed_attention_t6_t8(self):
    comp = self._check("hca", ("compressor/pool/position_bias", "compressor/pool/gate_proj/kernel"))
    attn = comp.bind(comp.params)
    hidden = comp.inputs()["hidden"]
    docs = dsv4._documents(None, *hidden.shape[:2])  # pylint: disable=protected-access
    q_res = attn.q_a_norm(attn.q_a_proj(hidden))
    compressed, _, _ = attn.compressor(hidden, q_res, docs)
    ok, metrics = layer_parity.compare(np.asarray(compressed), GOLDEN["attn_hca/diag/compressed_kv"][:, 0], _TOL)
    self.assertTrue(ok, metrics)

  def test_compressed_sparse_attention_t7_t8(self):
    comp = self._check("csa", ("compressor/pool/position_bias", "compressor/pool/kv_proj/kernel"))
    self.assertTrue(comp.spec["zero_grad_params"], "the reference indexer must have no gradient path")
    self.assertEqual(len(_reference_topk_tie_rows(GOLDEN["attn_csa/diag/index_scores"][0], 4, 2)), 0)
    # Selection is discrete, so the indexer's own negative control is dense selection: keeping every window must
    # change the output.
    dense = _Component("attn_csa", _attention_module(dataclasses.replace(self.cfg, index_topk=64), "csa"), self.cfg)
    ok, _ = layer_parity.compare(
        np.asarray(dense.bind(dense.params)(comp.inputs()["hidden"])[0]), GOLDEN["attn_csa/out/out"], _TOL
    )
    self.assertFalse(ok, "top-k sparse selection does not affect the output")
    attn = comp.bind(comp.params)
    hidden = comp.inputs()["hidden"]
    docs = dsv4._documents(None, *hidden.shape[:2])  # pylint: disable=protected-access
    q_res = attn.q_a_norm(attn.q_a_proj(hidden))
    compressed, _, diag = attn.compressor(hidden, q_res, docs)
    ok, metrics = layer_parity.compare(np.asarray(compressed), GOLDEN["attn_csa/diag/compressed_kv"][:, 0], _TOL)
    self.assertTrue(ok, metrics)
    np.testing.assert_array_equal(
        np.sort(np.asarray(diag["topk_indices"]), axis=-1), np.sort(GOLDEN["attn_csa/diag/topk_indices"], axis=-1)
    )

  def test_indexer_loss_is_opt_in_and_trains_only_the_indexer(self):
    comp = self._component("csa")
    hidden = comp.inputs()["hidden"]
    self.assertEqual(comp.bind(comp.params)(hidden)[1], {})
    loss_comp = _Component("attn_csa", _attention_module(_config(indexer_loss_scaling_factor=0.5), "csa"), self.cfg)

    def loss(p):
      return loss_comp.bind(p)(hidden)[1]["csa_indexer_loss"]

    value, grads = jax.value_and_grad(loss)(loss_comp.params)
    self.assertTrue(np.isfinite(float(value)) and float(value) > 0.0, value)
    indexer_norm = sum(float(jnp.sum(g**2)) for g in jax.tree.leaves(grads["compressor"]["indexer"]))
    self.assertGreater(indexer_norm, 0.0)
    for path in ("q_b_proj/kernel", "kv_proj/kernel", "compressor/pool/kv_proj/kernel"):
      self.assertEqual(float(jnp.max(jnp.abs(_leaf(grads, path)))), 0.0, path)

    # Short sequences (`1 <= seq < compress_rate_csa`) produce zero complete windows yet must preserve the PyTree keyset.
    short_hidden = hidden[:, : self.cfg.compress_rate_csa - 1]
    self.assertEqual(comp.bind(comp.params)(short_hidden)[1], {})
    short_out, short_aux = loss_comp.bind(loss_comp.params)(short_hidden)
    self.assertEqual(short_out.shape, short_hidden.shape)
    self.assertEqual(set(short_aux), {"csa_indexer_loss"})
    self.assertEqual(short_aux["csa_indexer_loss"].dtype, jnp.float32)
    self.assertEqual(float(short_aux["csa_indexer_loss"]), 0.0)

  def test_bfloat16_execution_precision_storage_and_parity(self):
    bf16_cfg = dsv4.DeepseekV4Config.from_hf_config(
        MANIFEST["hf_config"], dtype="bfloat16", weight_dtype="bfloat16", indexer_loss_scaling_factor=0.5
    )
    bf16_tol = layer_parity.Tolerance(atol=4e-2, rtol=4e-2, frob_rel=4e-2)

    norm = dsv4.DeepseekV4RMSNorm(
        bf16_cfg.hidden_size, eps=bf16_cfg.rms_norm_eps, param_dtype=bf16_cfg.param_dtype, sharding=None, name="norm"
    )
    self.assertEqual(norm.weight[...].dtype, jnp.bfloat16)
    norm_comp = _Component("rmsnorm", norm, bf16_cfg)
    x_bf16 = norm_comp.inputs()["x"].astype(jnp.bfloat16)
    y_norm, g_norm = jax.jit(jax.value_and_grad(lambda p: jnp.sum(norm_comp.bind(p)(x_bf16).astype(jnp.float32))))(
        norm_comp.params
    )
    self.assertTrue(np.isfinite(float(y_norm)))
    self.assertEqual(norm_comp.bind(norm_comp.params)(x_bf16).dtype, jnp.bfloat16)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(leaf.astype(jnp.float32)))) for leaf in jax.tree.leaves(g_norm)))

    hc = dsv4.DeepseekV4HyperConnection(bf16_cfg, sharding=None, rngs=nnx.Rngs(0), name="attn_hc")
    head = dsv4.DeepseekV4HyperHead(bf16_cfg, sharding=None, rngs=nnx.Rngs(0), name="hc_head")
    for leaf in (*jax.tree.leaves(nnx.state(hc)), *jax.tree.leaves(nnx.state(head))):
      self.assertEqual(leaf.dtype, jnp.float32)
    hc_comp = _Component("hyper_connection", hc, bf16_cfg)
    streams_bf16 = hc_comp.inputs()["x"].astype(jnp.bfloat16)
    post, comb, collapsed = hc_comp.bind(hc_comp.params)(streams_bf16)
    self.assertEqual((post.dtype, comb.dtype, collapsed.dtype), (jnp.float32, jnp.float32, jnp.bfloat16))
    head_comp = _Component("hyper_head", head, bf16_cfg)
    self.assertEqual(head_comp.bind(head_comp.params)(streams_bf16).dtype, jnp.bfloat16)

    for kind in ("sliding", "hca", "csa"):
      with self.subTest(kind=kind):
        mod = _attention_module(bf16_cfg, kind)
        self.assertEqual(mod.q_a_proj.kernel[...].dtype, jnp.bfloat16)
        self.assertEqual(mod.q_a_norm.weight[...].dtype, jnp.bfloat16)
        self.assertEqual(mod.sinks[...].dtype, jnp.float32)
        if mod.compressor is not None:
          self.assertEqual(mod.compressor.pool.position_bias[...].dtype, jnp.float32)
        comp_bf16 = _Component(f"attn_{kind}", mod, bf16_cfg)
        h_bf16 = comp_bf16.inputs()["hidden"].astype(jnp.bfloat16)
        out_eager, aux_eager = comp_bf16.bind(comp_bf16.params)(h_bf16)
        out_jit, aux_jit = nnx.jit(lambda m, x: m(x))(comp_bf16.bind(comp_bf16.params), h_bf16)
        self.assertEqual((out_eager.dtype, out_jit.dtype), (jnp.bfloat16, jnp.bfloat16))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out_eager.astype(jnp.float32)))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out_jit.astype(jnp.float32)))))
        if kind == "csa":
          self.assertEqual(aux_jit["csa_indexer_loss"].dtype, jnp.float32)
          self.assertTrue(np.isfinite(float(aux_jit["csa_indexer_loss"])))
        else:
          self.assertEqual((aux_eager, aux_jit), ({}, {}))

        def step_loss(p, x, c=comp_bf16):
          y, aux = c.bind(p)(x)
          total = jnp.mean(y.astype(jnp.float32) ** 2)
          for v in aux.values():
            total = total + v
          return total

        loss_val, (p_grads, x_grad) = jax.jit(jax.value_and_grad(step_loss, argnums=(0, 1)))(comp_bf16.params, h_bf16)
        self.assertTrue(np.isfinite(float(loss_val)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(x_grad.astype(jnp.float32)))))
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(g.astype(jnp.float32)))) for g in jax.tree.leaves(p_grads)))
        if kind in ("sliding", "hca"):
          ok, metrics = layer_parity.compare(
              np.asarray(out_jit, dtype=np.float32), GOLDEN[f"attn_{kind}/out/out"], bf16_tol
          )
          self.assertTrue(ok, f"{kind} bf16 vs fp32 golden: {metrics}")


def _reference_topk_tie_rows(scores: np.ndarray, rate: int, topk: int) -> np.ndarray:
  """Rows of a standalone reference [S, n] index-score matrix whose k-th and (k+1)-th visible scores are within 8 ULP."""
  ties = []
  for row in range(scores.shape[0]):
    visible = np.sort(scores[row, : (row + 1) // rate])[::-1]
    if visible.size <= topk:
      continue
    kth = np.float32(visible[topk - 1])
    margin = 8.0 * float(np.spacing(np.abs(kth)))
    if kth - visible[topk] <= margin:
      ties.append(row)
  return np.asarray(ties, dtype=np.int64)


class PackedDocumentsTest(ParityTestBase):
  """Packed documents at unaligned offsets match the reference run on each document alone."""

  def setUp(self):
    super().setUp()
    self.cfg = _config()

  def _run(self, comp_name: str, use_segments: bool, hidden_override: np.ndarray | None = None):
    spec = MANIFEST["components"][comp_name]
    kind = spec["param_source"].removeprefix("attn_")
    comp = _Component(comp_name, _attention_module(self.cfg, kind), self.cfg)
    hidden = _hashed(spec["input_key"], spec["inputs"][0]) if hidden_override is None else hidden_override
    seg = jnp.asarray(spec["segment_ids"], jnp.int32)[None] if use_segments else None
    return spec, hidden, np.asarray(comp.bind(comp.params)(jnp.asarray(hidden), seg)[0])

  def _packed_components(self) -> list[str]:
    names = sorted(name for name, spec in MANIFEST["components"].items() if "docs" in spec)
    self.assertEqual(len(names), 5)
    return names

  def test_packed_documents_match_standalone_reference(self):
    for name in self._packed_components():
      spec, _, out = self._run(name, use_segments=True)
      for doc_id, (start, stop) in enumerate(spec["docs"], start=1):
        rows = np.arange(stop - start)
        score_key = f"{name}/diag/index_scores_doc{doc_id}"
        if score_key in GOLDEN:
          ties = _reference_topk_tie_rows(GOLDEN[score_key][0], self.cfg.compress_rate_csa, self.cfg.index_topk)
          # Rows whose reference top-k is an exact or sub-8-ULP tie have an implementation-defined selection.
          self.assertLessEqual(len(ties), 2, f"{name} doc{doc_id}: too many reference ties {ties}")
          rows = np.setdiff1d(rows, ties)
        ok, metrics = layer_parity.compare(out[:, start:stop][:, rows], GOLDEN[f"{name}/out/doc{doc_id}"][:, rows], _TOL)
        self.assertTrue(ok, f"{name} doc{doc_id}: {metrics}")

  def test_negative_control_without_segment_ids_fails(self):
    for name in self._packed_components():
      spec, _, out = self._run(name, use_segments=False)
      start, stop = spec["docs"][1]
      ok, _ = layer_parity.compare(out[:, start:stop], GOLDEN[f"{name}/out/doc2"], _TOL)
      self.assertFalse(ok, f"{name}: ignoring document boundaries still matched the reference")

  def test_padding_and_other_documents_do_not_leak(self):
    for name in self._packed_components():
      spec, hidden, out = self._run(name, use_segments=True)
      seg = np.asarray(spec["segment_ids"])
      edited = hidden.copy()
      edited[0, seg != 2] += 3.0
      _, _, out_edited = self._run(name, use_segments=True, hidden_override=edited)
      start, stop = spec["docs"][1]
      np.testing.assert_array_equal(out_edited[:, start:stop], out[:, start:stop], err_msg=name)


class ShardingTest(ParityTestBase):
  """Sharded construction and execution on 4 CPU devices (CI sets 4 host devices)."""

  def setUp(self):
    super().setUp()
    self.cfg = _config()
    self.assertGreaterEqual(jax.device_count(), 4, "set XLA_FLAGS=--xla_force_host_platform_device_count=4")
    self.enterContext(nnx.use_eager_sharding(True))

  def _mesh(self, shape, axis_type=jax.sharding.AxisType.Auto):
    return jax.make_mesh(
        shape,
        ("data", "fsdp", "tensor", "expert"),
        axis_types=(axis_type,) * 4,
        devices=jax.devices()[:4],
    )

  def test_sharded_attention_matches_golden(self):
    for shape in ((1, 2, 2, 1), (1, 4, 1, 1), (1, 1, 2, 2)):
      for kind in ("sliding", "csa", "hca"):
        with self.subTest(mesh=shape, kind=kind), jax.sharding.set_mesh(self._mesh(shape)):
          comp = _Component(f"attn_{kind}", _attention_module(self.cfg, kind, dsv4.DeepseekV4Sharding()), self.cfg)
          sharded = [
              path
              for path, leaf in jax.tree_util.tree_flatten_with_path(comp.params)[0]
              if leaf.ndim >= 2 and leaf.size >= 256 and not leaf.sharding.is_fully_replicated
          ]
          large = [leaf for leaf in jax.tree.leaves(comp.params) if leaf.ndim >= 2 and leaf.size >= 256]
          self.assertEqual(len(sharded), len(large), f"replicated large parameters on mesh {shape}")
          out = nnx.jit(lambda m, h: m(h)[0])(comp.bind(comp.params), comp.inputs()["hidden"])
          ok, metrics = layer_parity.compare(np.asarray(out), GOLDEN[f"attn_{kind}/out/out"], _TOL)
          self.assertTrue(ok, metrics)

  def test_sharded_gradient_parity_tp1_vs_tp2(self):
    for shape in ((1, 4, 1, 1), (1, 2, 2, 1), (1, 1, 2, 2)):
      for kind in ("sliding", "csa", "hca"):
        with self.subTest(mesh=shape, kind=kind), jax.sharding.set_mesh(self._mesh(shape)):
          comp = _Component(f"attn_{kind}", _attention_module(self.cfg, kind, dsv4.DeepseekV4Sharding()), self.cfg)
          self.assert_parity(comp, lambda p, inp, c=comp: {"out": c.bind(p)(inp["hidden"])[0]})

  def test_param_spec_trims_mesh_axes_that_do_not_divide(self):
    sharding = dsv4.DeepseekV4Sharding()
    mesh = jax.sharding.AbstractMesh((1, 64, 1, 8), ("data", "fsdp", "tensor", "expert"))
    with jax.sharding.use_abstract_mesh(mesh):
      self.assertEqual(
          sharding.param_spec("indexer.position_bias", (None, "compress_feature"), (4, 256)),
          jax.sharding.PartitionSpec(None, "fsdp"),
      )
      self.assertEqual(
          sharding.param_spec("kv_proj", ("embed", None), (7168, 512)),
          jax.sharding.PartitionSpec(("fsdp", "expert"), None),
      )
    wide = jax.sharding.AbstractMesh((1, 64, 1, 32), ("data", "fsdp", "tensor", "expert"))
    with jax.sharding.use_abstract_mesh(wide):
      self.assertEqual(
          sharding.param_spec("kv_proj", ("embed", None), (7168, 512)), jax.sharding.PartitionSpec("fsdp", None)
      )
      self.assertEqual(
          sharding.param_spec("o_a_proj", ("o_groups", "o_in", None), (16, 4096, 1024)),
          jax.sharding.PartitionSpec("tensor", ("fsdp", "expert"), None),
      )
    with self.assertRaisesRegex(ValueError, "no mesh"):
      sharding.param_spec("kv_proj", ("embed", None), (7168, 512))

  def test_module_built_under_mesh_runs_outside_mesh(self):
    with jax.sharding.set_mesh(self._mesh((1, 2, 2, 1))):
      comp = _Component("attn_csa", _attention_module(self.cfg, "csa", dsv4.DeepseekV4Sharding()), self.cfg)
      hidden = comp.inputs()["hidden"]
    self.assertTrue(jax.sharding.get_abstract_mesh().empty)
    out_eager, _ = comp.bind(comp.params)(hidden)
    out_jit = nnx.jit(lambda m, h: m(h)[0])(comp.bind(comp.params), hidden)
    for tag, out in (("eager", out_eager), ("jit", out_jit)):
      ok, metrics = layer_parity.compare(np.asarray(out), GOLDEN["attn_csa/out/out"], _TOL)
      self.assertTrue(ok, f"{tag} outside mesh: {metrics}")

  def test_explicit_mesh_attention_and_compressor_forward_and_grad(self):
    chk_cfg = dataclasses.replace(
        self.cfg,
        attn_query_chunk_size=7,
        index_query_chunk_size=5,
        index_window_chunk_size=2,
        index_head_chunk_size=2,
        indexer_loss_scaling_factor=0.5,
    )
    rng = np.random.default_rng(7)
    hidden_np = rng.standard_normal((4, 20, chk_cfg.hidden_size)).astype(np.float32) * 0.5
    seg_np = np.asarray(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
        ],
        dtype=np.int32,
    )
    # Build unsharded reference parameters, outputs, and gradients for each layer kind and segment mode.
    refs = {}
    for kind in ("sliding", "csa", "hca"):
      ref_mod = _attention_module(chk_cfg, kind, sharding=None)
      gd_ref, st_ref = nnx.split(ref_mod)
      p_ref = nnx.to_pure_dict(st_ref)
      for use_seg in (False, True):

        def ref_loss(p_in, x_in, s_in, g=gd_ref, s=st_ref):
          st_c = jax.tree.map(lambda leaf: leaf, s)
          nnx.replace_by_pure_dict(st_c, p_in)
          out, aux = nnx.merge(g, st_c)(x_in, s_in)
          loss = jnp.sum(out**2) + sum(aux.values(), jnp.zeros((), dtype=jnp.float32))
          return loss, out

        seg_arg = jnp.asarray(seg_np) if use_seg else None
        (val_ref, out_ref), (pg_ref, xg_ref) = jax.jit(jax.value_and_grad(ref_loss, argnums=(0, 1), has_aux=True))(
            p_ref, jnp.asarray(hidden_np), seg_arg
        )
        refs[(kind, use_seg)] = (
            p_ref,
            float(val_ref),
            np.asarray(out_ref),
            jax.tree.map(np.asarray, pg_ref),
            np.asarray(xg_ref),
        )

    sh = dsv4.DeepseekV4Sharding()
    for shape in ((1, 2, 2, 1), (1, 4, 1, 1), (1, 1, 2, 2), (2, 1, 2, 1)):
      with jax.sharding.set_mesh(self._mesh(shape, jax.sharding.AxisType.Explicit)):
        x_sh = sh.constrain(jnp.asarray(hidden_np), "hidden", ("batch", "length", "embed"))
        seg_sh = sh.constrain(jnp.asarray(seg_np), "segment_ids", ("batch", "length"))

        # Standalone DeepseekV4Compressor (CSA retain_scores=True and HCA) with batch-sharded packed segment_ids.
        for kind, ltype in (("csa", dsv4.COMPRESSED_SPARSE_ATTENTION), ("hca", dsv4.HEAVILY_COMPRESSED_ATTENTION)):
          with self.subTest(mesh=shape, module="compressor", kind=kind):
            comp_mod = dsv4.DeepseekV4Compressor(chk_cfg, ltype, sharding=sh, rngs=nnx.Rngs(0), name=f"comp_{kind}")
            c_gd, c_st = nnx.split(comp_mod)
            c_p = nnx.to_pure_dict(c_st)
            q_res = jnp.ones((4, 20, chk_cfg.q_lora_rank), dtype=jnp.float32) * 0.2
            q_res = sh.constrain(q_res, "q_res", ("batch", "length", "q_lora"))

            def comp_loss(p_in, x_in, qr_in, s_in, g=c_gd, s=c_st):
              st_c = jax.tree.map(lambda leaf: leaf, s)
              nnx.replace_by_pure_dict(st_c, p_in)
              docs = dsv4._documents(s_in, *x_in.shape[:2])  # pylint: disable=protected-access
              c_out, sel_mask, _ = nnx.merge(g, st_c)(x_in, qr_in, docs, return_mask=True)
              return jnp.sum(c_out**2) + jnp.sum(sel_mask.astype(jnp.float32)) * 0.0

            cv, (cpg, cxg) = jax.jit(jax.value_and_grad(comp_loss, argnums=(0, 1)))(c_p, x_sh, q_res, seg_sh)
            self.assertTrue(np.isfinite(float(cv)))
            self.assertTrue(bool(jnp.all(jnp.isfinite(cxg))))
            self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(cpg)))

        # Full DeepseekV4Attention across all 3 layer types, with and without packed segment_ids.
        for kind in ("sliding", "csa", "hca"):
          for use_seg in (False, True):
            with self.subTest(mesh=shape, module="attention", kind=kind, packed=use_seg):
              p_ref, val_ref, out_ref, pg_ref, xg_ref = refs[(kind, use_seg)]
              mod = _attention_module(chk_cfg, kind, sh)
              gd, st = nnx.split(mod)
              p_sh = jax.tree.map(
                  lambda ref_leaf, sh_leaf: jax.device_put(ref_leaf, sh_leaf.sharding), p_ref, nnx.to_pure_dict(st)
              )

              def exp_loss(p_in, x_in, s_in, g=gd, s=st):
                st_c = jax.tree.map(lambda leaf: leaf, s)
                nnx.replace_by_pure_dict(st_c, p_in)
                out, aux = nnx.merge(g, st_c)(x_in, s_in)
                loss = jnp.sum(out**2) + sum(aux.values(), jnp.zeros((), dtype=jnp.float32))
                return loss, out

              seg_arg = seg_sh if use_seg else None
              (val_sh, out_sh), (pg_sh, xg_sh) = jax.jit(jax.value_and_grad(exp_loss, argnums=(0, 1), has_aux=True))(
                  p_sh, x_sh, seg_arg
              )
              self.assertAlmostEqual(float(val_sh), val_ref, places=4)
              ok_o, metrics_o = layer_parity.compare(np.asarray(out_sh), out_ref, _TOL)
              self.assertTrue(ok_o, f"explicit out ({shape}, {kind}, seg={use_seg}): {metrics_o}")
              ok_x, metrics_x = layer_parity.compare(np.asarray(xg_sh), xg_ref, _TOL)
              self.assertTrue(ok_x, f"explicit x_grad ({shape}, {kind}, seg={use_seg}): {metrics_x}")
              for (pth, gs), (_, gr) in zip(
                  jax.tree_util.tree_flatten_with_path(pg_sh)[0], jax.tree_util.tree_flatten_with_path(pg_ref)[0]
              ):
                ok_g, metrics_g = layer_parity.compare(np.asarray(gs), gr, _TOL)
                self.assertTrue(ok_g, f"explicit param grad {pth} ({shape}, {kind}, seg={use_seg}): {metrics_g}")

    # Edge cases on Explicit (1, 2, 2, 1): batch=2 (< data*fsdp*expert) golden parity, seq=0, seq=2, bfloat16, and mHC.
    with jax.sharding.set_mesh(self._mesh((1, 2, 2, 1), jax.sharding.AxisType.Explicit)):
      for kind in ("sliding", "csa", "hca"):
        with self.subTest(edge="batch2_golden", kind=kind):
          comp = _Component(f"attn_{kind}", _attention_module(self.cfg, kind, sh), self.cfg)
          self.assert_parity(comp, lambda p, inp, c=comp: {"out": c.bind(p)(inp["hidden"])[0]})
      for seq_len in (0, 2):
        for kind in ("sliding", "csa", "hca"):
          with self.subTest(edge=f"seq{seq_len}", kind=kind):
            mod = _attention_module(chk_cfg, kind, sh)
            gd, st = nnx.split(mod)
            p_sh = nnx.to_pure_dict(st)
            x_e = sh.constrain(
                jnp.ones((4, seq_len, chk_cfg.hidden_size), dtype=jnp.float32) * 0.1, "h", ("batch", "length", "embed")
            )
            s_e = sh.constrain(jnp.ones((4, seq_len), dtype=jnp.int32), "s", ("batch", "length"))
            for s_arg in (None, s_e):

              def edge_loss(p_in, x_in, s_in, g=gd, s=st):
                st_c = jax.tree.map(lambda leaf: leaf, s)
                nnx.replace_by_pure_dict(st_c, p_in)
                out, aux = nnx.merge(g, st_c)(x_in, s_in)
                return jnp.sum(out) + sum(aux.values(), jnp.zeros((), dtype=jnp.float32))

              ev, (epg, exg) = jax.jit(jax.value_and_grad(edge_loss, argnums=(0, 1)))(p_sh, x_e, s_arg)
              self.assertTrue(np.isfinite(float(ev)))
              self.assertTrue(bool(jnp.all(jnp.isfinite(exg))))
              self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(epg)))
      bf16_cfg = dsv4.DeepseekV4Config.from_hf_config(
          MANIFEST["hf_config"],
          dtype="bfloat16",
          weight_dtype="bfloat16",
          attn_query_chunk_size=7,
          index_query_chunk_size=5,
          index_window_chunk_size=2,
          index_head_chunk_size=2,
          indexer_loss_scaling_factor=0.5,
      )
      for kind in ("csa", "hca"):
        with self.subTest(edge="bfloat16", kind=kind):
          mod = _attention_module(bf16_cfg, kind, sh)
          gd, st = nnx.split(mod)
          x_bf = sh.constrain(
              jnp.ones((4, 20, bf16_cfg.hidden_size), dtype=jnp.bfloat16) * 0.1, "h", ("batch", "length", "embed")
          )

          def bf_loss(p_in, x_in, g=gd, s=st):
            st_c = jax.tree.map(lambda leaf: leaf, s)
            nnx.replace_by_pure_dict(st_c, p_in)
            out, aux = nnx.merge(g, st_c)(x_in, None)
            return jnp.sum(out.astype(jnp.float32)) + sum(aux.values(), jnp.zeros((), dtype=jnp.float32))

          bv, (bpg, bxg) = jax.jit(jax.value_and_grad(bf_loss, argnums=(0, 1)))(nnx.to_pure_dict(st), x_bf)
          self.assertTrue(np.isfinite(float(bv)))
          self.assertTrue(bool(jnp.all(jnp.isfinite(bxg))))
          self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(bpg)))
      for mod_cls, name in ((dsv4.DeepseekV4HyperConnection, "hc"), (dsv4.DeepseekV4HyperHead, "hh")):
        with self.subTest(edge=name):
          hm = mod_cls(chk_cfg, sharding=sh, rngs=nnx.Rngs(0), name=name)
          h_gd, h_st = nnx.split(hm)
          x_hc = jnp.ones((4, 20, chk_cfg.hc_mult, chk_cfg.hidden_size), dtype=jnp.float32) * 0.1
          x_hc = sh.constrain(x_hc, "hc_in", ("batch", "length", None, "embed"))

          def hc_loss(p_in, x_in, g=h_gd, s=h_st):
            st_c = jax.tree.map(lambda leaf: leaf, s)
            nnx.replace_by_pure_dict(st_c, p_in)
            res = nnx.merge(g, st_c)(x_in)
            return sum(jnp.sum(r) for r in res) if isinstance(res, tuple) else jnp.sum(res)

          hv, (hpg, hxg) = jax.jit(jax.value_and_grad(hc_loss, argnums=(0, 1)))(nnx.to_pure_dict(h_st), x_hc)
          self.assertTrue(np.isfinite(float(hv)))
          self.assertTrue(bool(jnp.all(jnp.isfinite(hxg))))
          self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(hpg)))


class Float64AlgebraicEquivalenceTest(ParityTestBase):
  """Row T14: float64 forward and all-leaf gradient parity (frob_rel <= 1e-10) under jax_enable_x64."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "cpu":
      self.skipTest("IEEE float64 T14 algebraic-equivalence check runs on CPU")
    self.enterContext(jax.enable_x64(True))
    self.cfg = _f64_config()

  def test_norms_rope_grouped_and_mhc_float64_t14(self):
    rmsnorm = dsv4.DeepseekV4RMSNorm(
        self.cfg.hidden_size, eps=self.cfg.rms_norm_eps, param_dtype=jnp.float64, sharding=None, name="norm"
    )
    comp_norm = _Component("f64_rmsnorm", rmsnorm, self.cfg)
    self.assert_parity(comp_norm, lambda p, inp: {"y": comp_norm.bind(p)(inp["x"])}, tol=_FP64_TOL)

    comp_unw = _Component("f64_unweighted_rmsnorm", None, self.cfg)
    self.assert_parity(
        comp_unw, lambda p, inp: {"y": dsv4.unweighted_rms_norm(inp["x"], self.cfg.rms_norm_eps)}, tol=_FP64_TOL
    )

    for rope_type in ("main", "compress"):
      inv64 = dsv4.rope_inv_freq(self.cfg, rope_type, dtype=np.float64)
      self.assertEqual(inv64.dtype, np.float64)
      ok, metrics = layer_parity.compare(inv64, GOLDEN[f"f64_rope_tables/out/{rope_type}_inv_freq"], _FP64_TOL)
      self.assertTrue(ok, f"f64 {rope_type} inv_freq: {metrics}")
      start, stop = MANIFEST["components"]["f64_rope_tables"]["positions"]
      cos, sin = dsv4.rope_cos_sin(tuple(inv64), jnp.arange(start, stop)[None], jnp.float64)
      for name, value in (("cos", cos), ("sin", sin)):
        ok, metrics = layer_parity.compare(
            np.asarray(value), GOLDEN[f"f64_rope_tables/out/{rope_type}_{name}"], _FP64_TOL
        )
        self.assertTrue(ok, f"f64 {rope_type} {name}: {metrics}")

    comp_rope = _Component("f64_rope_apply", None, self.cfg)

    def rope_fn(_, inp):
      x = inp["x"]
      batch, _, seq, _ = x.shape
      pos = jnp.broadcast_to(jnp.arange(seq), (batch, seq))
      x_bshd = jnp.swapaxes(x, 1, 2)
      out = {}
      for rope_type in ("main", "compress"):
        inv64 = dsv4.rope_inv_freq(self.cfg, rope_type, dtype=np.float64)
        cos, sin = dsv4.rope_cos_sin(tuple(inv64), pos, x.dtype)
        out[f"{rope_type}_rotated"] = jnp.swapaxes(dsv4.apply_rotary(x_bshd, cos, sin), 1, 2)
        out[f"{rope_type}_inverse"] = jnp.swapaxes(dsv4.apply_rotary(x_bshd, cos, -sin), 1, 2)
      return out

    self.assert_parity(comp_rope, rope_fn, tol=_FP64_TOL)

    in_g = self.cfg.num_attention_heads * self.cfg.head_dim // self.cfg.o_groups
    grouped = dsv4.DeepseekV4GroupedLinear(
        self.cfg, in_g, self.cfg.o_lora_rank, self.cfg.o_groups, sharding=None, rngs=nnx.Rngs(0), name="o_a_proj"
    )
    comp_grp = _Component("f64_grouped_linear", grouped, self.cfg, path_fn=lambda name: ("weight", "G"))
    self.assert_parity(comp_grp, lambda p, inp: {"y": comp_grp.bind(p)(inp["x"])}, tol=_FP64_TOL)

    hyper = dsv4.DeepseekV4HyperConnection(self.cfg, sharding=None, rngs=nnx.Rngs(0), name="attn_hc")
    comp_hc = _Component("f64_hyper_connection", hyper, self.cfg)

    def hc_fn(p, inp):
      post, comb, collapsed = comp_hc.bind(p)(inp["x"])
      return {"post": post, "comb": comb, "collapsed": collapsed}

    self.assert_parity(comp_hc, hc_fn, tol=_FP64_TOL)

    head = dsv4.DeepseekV4HyperHead(self.cfg, sharding=None, rngs=nnx.Rngs(0), name="hc_head")
    comp_head = _Component("f64_hyper_head", head, self.cfg)
    self.assert_parity(comp_head, lambda p, inp: {"y": comp_head.bind(p)(inp["x"])}, tol=_FP64_TOL)

  def test_attention_all_types_float64_t14(self):
    for kind in ("sliding", "csa", "hca"):
      with self.subTest(kind=kind):
        comp = _Component(f"f64_attn_{kind}", _attention_module(self.cfg, kind), self.cfg)
        self.assert_parity(comp, lambda p, inp, c=comp: {"out": c.bind(p)(inp["hidden"])[0]}, tol=_FP64_TOL)


class ConfigTest(unittest.TestCase):
  """Verifies DeepSeek-V4 Pro/Flash config parsing and validation guards."""

  def _hf(self, **overrides) -> dict[str, Any]:
    hf = {k: v for k, v in MANIFEST["hf_config"].items() if k not in ("layer_types", "compress_rates")}
    hf.update(overrides)
    return hf

  def test_pro_schedule_comes_from_compress_ratios(self):
    ratios = [128, 128] + [4, 128] * 29 + [4] + [0, 0, 0]
    cfg = dsv4.DeepseekV4Config.from_hf_config(
        self._hf(num_hidden_layers=61, compress_ratios=ratios), dtype="float32", weight_dtype="float32"
    )
    hca, csa = dsv4.HEAVILY_COMPRESSED_ATTENTION, dsv4.COMPRESSED_SPARSE_ATTENTION
    self.assertEqual(cfg.layer_types[:4], (hca, hca, csa, hca))
    self.assertEqual(cfg.layer_types[-1], csa)
    self.assertEqual((cfg.layer_types.count(hca), cfg.layer_types.count(csa)), (31, 30))
    self.assertEqual((cfg.compress_rate_csa, cfg.compress_rate_hca), (4, 128))

  def test_ignores_unmapped_trailing_draft_compress_ratios(self):
    ratios_with_draft = [128, 128] + [4, 128] * 29 + [4] + [999, -1]
    cfg = dsv4.DeepseekV4Config.from_hf_config(
        self._hf(num_hidden_layers=61, compress_ratios=ratios_with_draft), dtype="float32", weight_dtype="float32"
    )
    self.assertEqual(len(cfg.layer_types), 61)
    self.assertEqual(cfg.layer_types[-1], dsv4.COMPRESSED_SPARSE_ATTENTION)
    with self.assertRaisesRegex(ValueError, r"compress_ratios has values \[999\]"):
      dsv4.DeepseekV4Config.from_hf_config(
          self._hf(num_hidden_layers=61, compress_ratios=[999] + ratios_with_draft[1:]),
          dtype="float32",
          weight_dtype="float32",
      )

  def test_flash_release_config_fields(self):
    hf = self._hf(
        num_hidden_layers=43,
        num_attention_heads=64,
        head_dim=512,
        index_head_dim=128,
        o_groups=8,
        compress_ratios=[0, 0, 4, 128] + [4, 128] * 19 + [4, 0],
        qk_rope_head_dim=64,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
    )
    del hf["partial_rotary_factor"], hf["rope_parameters"]
    cfg = dsv4.DeepseekV4Config.from_hf_config(hf, dtype="bfloat16", weight_dtype="bfloat16")
    self.assertEqual(cfg.layer_types[:3], (dsv4.SLIDING_ATTENTION,) * 2 + (dsv4.COMPRESSED_SPARSE_ATTENTION,))
    self.assertEqual(
        (cfg.qk_rope_head_dim, cfg.indexer_loss_scaling_factor, cfg.megablox, cfg.hardware), (64, 0.0, True, "tpu")
    )

  def test_ingests_transformers_v5_nested_rope_and_hf_model_configs(self):
    nested_v5 = self._hf(
        layer_types=MANIFEST["hf_config"]["layer_types"],
        rope_scaling=None,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
            "compress": {
                "rope_type": "yarn",
                "rope_theta": 160000.0,
                "partial_rotary_factor": 0.25,
                "factor": 16.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096,
                "attention_factor": 1.0,
            },
        },
    )
    del nested_v5["rope_theta"], nested_v5["compress_rope_theta"], nested_v5["partial_rotary_factor"]
    cfg_v5 = dsv4.DeepseekV4Config.from_hf_config(nested_v5, dtype="bfloat16", weight_dtype="bfloat16")
    self.assertEqual((cfg_v5.qk_rope_head_dim, cfg_v5.rope_theta, cfg_v5.compress_rope_theta), (8, 10000.0, 160000.0))

    standardized = self._hf(
        layer_types=MANIFEST["hf_config"]["layer_types"],
        qk_rope_head_dim=8,
        partial_rotary_factor=None,
        rope_theta=None,
        rope_scaling=None,
        rope_parameters={
            "rope_type": "yarn",
            "type": "yarn",
            "rope_theta": 10000.0,
            "factor": 16.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096,
        },
    )
    cfg_std = dsv4.DeepseekV4Config.from_hf_config(standardized, dtype="bfloat16", weight_dtype="bfloat16")
    self.assertEqual((cfg_std.qk_rope_head_dim, cfg_std.rope_theta, cfg_std.compress_rope_theta), (8, 10000.0, 160000.0))

    # Exercise live MaxText HF_MODEL_CONFIGS when present outside minimal Bazel strict_deps unit targets.
    try:
      hf_cfgs = importlib.import_module("maxtext.checkpoint_conversion.utils.hf_model_configs").HF_MODEL_CONFIGS
    except ModuleNotFoundError:
      hf_cfgs = None
    if hf_cfgs is not None:
      for name, expected_layers in (("deepseek4-284b", 43), ("deepseek4-1600b", 61)):
        parsed = dsv4.DeepseekV4Config.from_hf_config(hf_cfgs[name].to_dict(), dtype="bfloat16", weight_dtype="bfloat16")
        self.assertEqual(
            (parsed.num_hidden_layers, parsed.qk_rope_head_dim, parsed.rope_theta), (expected_layers, 64, 10000.0)
        )

  def test_rejects_ambiguous_or_unsupported_configs(self):
    with self.assertRaisesRegex(ValueError, "refusing to guess"):
      dsv4.DeepseekV4Config.from_hf_config(self._hf(), dtype="float32", weight_dtype="float32")
    with self.assertRaisesRegex(ValueError, "one shared KV head"):
      dsv4.DeepseekV4Config.from_hf_config(
          self._hf(num_key_value_heads=2, layer_types=MANIFEST["hf_config"]["layer_types"]),
          dtype="float32",
          weight_dtype="float32",
      )
    bad_rope = dict(MANIFEST["hf_config"]["rope_parameters"], attention_factor=1.277)
    with self.assertRaisesRegex(ValueError, "attention_factor"):
      dsv4.DeepseekV4Config.from_hf_config(
          self._hf(rope_parameters=bad_rope, layer_types=MANIFEST["hf_config"]["layer_types"]),
          dtype="float32",
          weight_dtype="float32",
      )
    with self.assertRaisesRegex(ValueError, "entries"):
      dsv4.DeepseekV4Config.from_hf_config(self._hf(compress_ratios=[0, 0, 4]), dtype="float32", weight_dtype="float32")
    with self.assertRaisesRegex(ValueError, "index_head_dim"):
      dsv4.DeepseekV4Config.from_hf_config(
          self._hf(index_head_dim=4, layer_types=MANIFEST["hf_config"]["layer_types"]),
          dtype="float32",
          weight_dtype="float32",
      )
    for field in ("sliding_window", "hc_mult", "hc_sinkhorn_iters"):
      with self.assertRaisesRegex(ValueError, "must be >= 1"):
        dsv4.DeepseekV4Config.from_hf_config(
            self._hf(**{field: 0, "layer_types": MANIFEST["hf_config"]["layer_types"]}),
            dtype="float32",
            weight_dtype="float32",
        )
    for field in ("attn_query_chunk_size", "index_query_chunk_size", "index_window_chunk_size", "index_head_chunk_size"):
      with self.assertRaisesRegex(ValueError, "must all be >= 1"):
        dsv4.DeepseekV4Config.from_hf_config(
            self._hf(layer_types=MANIFEST["hf_config"]["layer_types"]),
            dtype="float32",
            weight_dtype="float32",
            **{field: 0},
        )


def _jaxpr_shapes(jaxpr: jax.extend.core.Jaxpr) -> list[tuple[int, ...]]:
  """Collects every array shape across `jaxpr` and all nested scan/remat/call sub-jaxprs."""
  shapes: list[tuple[int, ...]] = []
  seen: set[int] = set()

  def visit(jp: jax.extend.core.Jaxpr) -> None:
    if id(jp) in seen:
      return
    seen.add(id(jp))
    for var in (*jp.invars, *jp.constvars, *jp.outvars):
      aval = getattr(var, "aval", None)
      if hasattr(aval, "shape") and isinstance(aval.shape, tuple):
        shapes.append(tuple(int(d) for d in aval.shape if isinstance(d, int)))
    for eqn in jp.eqns:
      for var in (*eqn.invars, *eqn.outvars):
        aval = getattr(var, "aval", None)
        if hasattr(aval, "shape") and isinstance(aval.shape, tuple):
          shapes.append(tuple(int(d) for d in aval.shape if isinstance(d, int)))
      for val in eqn.params.values():
        if isinstance(val, jax.extend.core.ClosedJaxpr):
          visit(val.jaxpr)
        elif isinstance(val, jax.extend.core.Jaxpr):
          visit(val)
        elif isinstance(val, (tuple, list)):
          for item in val:
            if isinstance(item, jax.extend.core.ClosedJaxpr):
              visit(item.jaxpr)
            elif isinstance(item, jax.extend.core.Jaxpr):
              visit(item)

  visit(jaxpr)
  return shapes


_HLO_SHAPE_RE = re.compile(r"\b(?:f16|bf16|f32|f64|s32|u32|pred)\[([0-9,]+)\]")


def _compiled_hlo_shapes(fn: Callable[..., Any], *args: Any) -> list[tuple[int, ...]]:
  """Compiles `fn(*args)` with XLA and extracts every materialized tensor shape in compiled HLO."""
  hlo_text = jax.jit(fn).lower(*args).compile().as_text()
  return [tuple(int(dim) for dim in match.split(",")) for match in _HLO_SHAPE_RE.findall(hlo_text)]


def _has_exact_multiset(shapes: list[tuple[int, ...]], target_dims: tuple[int, ...]) -> bool:
  """True when any tensor has the exact dimension multiset `target_dims`."""
  req = sorted(target_dims)
  return any(sorted(shape) == req for shape in shapes)


class ChunkedIndexerTest(ParityTestBase):
  """Verifies chunked Lightning Indexer memory bounds (jaxpr + compiled HLO) and exact parity."""

  def setUp(self):
    super().setUp()
    self.cfg = _config()

  def test_scores_never_materializes_4d_bsht_tensor_in_jaxpr_or_compiled_hlo(self):
    batch, seq, heads, head_dim, rate = 2, 256, 32, 16, 4
    n_win = seq // rate  # 64 windows; seq=256 > 128 default query chunk and heads=32 > 16 default head chunk.

    # Positive control: prove both jaxpr and compiled XLA HLO detectors catch an unchunked [B, S, H, n] einsum.
    q_probe = jnp.ones((batch, seq, heads, head_dim), dtype=jnp.float32)
    k_probe = jnp.ones((batch, n_win, head_dim), dtype=jnp.float32)

    def unchunked_probe(q_in, k_in):
      return jnp.sum(jax.nn.relu(jnp.einsum("bshd,btd->bsht", q_in, k_in)), axis=2)

    probe_jp = jax.make_jaxpr(unchunked_probe)(q_probe, k_probe).jaxpr
    self.assertTrue(_has_exact_multiset(_jaxpr_shapes(probe_jp), (batch, seq, heads, n_win)))
    self.assertTrue(
        _has_exact_multiset(_compiled_hlo_shapes(unchunked_probe, q_probe, k_probe), (batch, seq, heads, n_win))
    )

    # Keep index_n_heads=32 > index_head_chunk_size=16 (distinct from num_attention_heads=8) so `_has_exact_multiset`
    # specifically targets the indexer's 4D [B, S, H_idx, n] score tensor.
    hf = dict(MANIFEST["hf_config"], index_n_heads=heads, index_head_dim=head_dim, index_topk=8)
    for loss_scale in (0.0, 0.5):
      with self.subTest(loss_scale=loss_scale):
        cfg = dsv4.DeepseekV4Config.from_hf_config(
            hf, dtype="float32", weight_dtype="float32", indexer_loss_scaling_factor=loss_scale
        )
        attn = _attention_module(cfg, "csa")
        graphdef, state = nnx.split(attn)
        pure_params = nnx.to_pure_dict(state)
        hidden = jnp.ones((batch, seq, cfg.hidden_size), dtype=jnp.float32) * 0.1

        def run_attn(p, x, gd=graphdef, st=state):
          st_copy = jax.tree.map(lambda leaf: leaf, st)
          nnx.replace_by_pure_dict(st_copy, p)
          mod = nnx.merge(gd, st_copy)
          out, aux = mod(x)
          loss = jnp.sum(out) + sum(aux.values(), jnp.zeros((), dtype=jnp.float32))
          return loss

        fwd_jp = jax.make_jaxpr(run_attn)(pure_params, hidden).jaxpr
        self.assertFalse(
            _has_exact_multiset(_jaxpr_shapes(fwd_jp), (batch, seq, heads, n_win)),
            f"Dense [B, S, H, n]={(batch, seq, heads, n_win)} materialized in forward jaxpr (loss_scale={loss_scale})",
        )
        self.assertFalse(
            _has_exact_multiset(_compiled_hlo_shapes(run_attn, pure_params, hidden), (batch, seq, heads, n_win)),
            f"Dense [B, S, H, n]={(batch, seq, heads, n_win)} materialized in compiled HLO (loss_scale={loss_scale})",
        )

        grad_fn = jax.value_and_grad(run_attn, argnums=(0, 1))
        bwd_jp = jax.make_jaxpr(grad_fn)(pure_params, hidden).jaxpr
        self.assertFalse(
            _has_exact_multiset(_jaxpr_shapes(bwd_jp), (batch, seq, heads, n_win)),
            f"Dense [B, S, H, n] materialized in value_and_grad jaxpr (loss_scale={loss_scale})",
        )
        self.assertFalse(
            _has_exact_multiset(_compiled_hlo_shapes(grad_fn, pure_params, hidden), (batch, seq, heads, n_win)),
            f"Dense [B, S, H, n] materialized in value_and_grad compiled HLO (loss_scale={loss_scale})",
        )

    # Also verify head-only overflow (short seq=96 <= query_chunk_size=128, heads=32 > 16) never allocates [B, S, H, n],
    # and streaming compressor (`return_mask=False`, loss_scale=0.0) avoids both [B, S, H, n] and [B, S, n].
    short_seq, short_n = 96, 24
    cfg_short = dsv4.DeepseekV4Config.from_hf_config(hf, dtype="float32", weight_dtype="float32")
    comp_mod = dsv4.DeepseekV4Compressor(
        cfg_short, dsv4.COMPRESSED_SPARSE_ATTENTION, sharding=None, rngs=nnx.Rngs(0), name="comp"
    )
    gd_s, st_s = nnx.split(comp_mod)
    p_s = nnx.to_pure_dict(st_s)
    h_short = jnp.ones((batch, short_seq, cfg_short.hidden_size), dtype=jnp.float32)
    qr_short = jnp.ones((batch, short_seq, cfg_short.q_lora_rank), dtype=jnp.float32)

    def run_short_scores(p, x, qr):
      st_copy = jax.tree.map(lambda leaf: leaf, st_s)
      nnx.replace_by_pure_dict(st_copy, p)
      mod = nnx.merge(gd_s, st_copy)
      docs = dsv4._documents(None, batch, short_seq)  # pylint: disable=protected-access
      win = dsv4._windows(docs, cfg_short.compress_rate_csa)  # pylint: disable=protected-access
      return mod.indexer.scores(x, qr, docs, win)

    short_hlo = _compiled_hlo_shapes(run_short_scores, p_s, h_short, qr_short)
    self.assertTrue(_has_exact_multiset(short_hlo, (batch, short_seq, short_n)))
    self.assertFalse(_has_exact_multiset(short_hlo, (batch, short_seq, heads, short_n)))

    cfg_stream = dataclasses.replace(
        cfg_short, index_query_chunk_size=16, index_window_chunk_size=8, index_head_chunk_size=8
    )
    comp_stream = dsv4.DeepseekV4Compressor(
        cfg_stream, dsv4.COMPRESSED_SPARSE_ATTENTION, sharding=None, rngs=nnx.Rngs(1), name="comp"
    )
    gd_st, st_st = nnx.split(comp_stream)
    p_st = nnx.to_pure_dict(st_st)

    def run_streaming_compressor(p, x, qr):
      st_copy = jax.tree.map(lambda leaf: leaf, st_st)
      nnx.replace_by_pure_dict(st_copy, p)
      mod = nnx.merge(gd_st, st_copy)
      docs = dsv4._documents(None, batch, short_seq)  # pylint: disable=protected-access
      compressed, _, diag = mod(x, qr, docs, return_mask=False)
      return compressed, diag["topk_indices"]

    stream_jp = jax.make_jaxpr(run_streaming_compressor)(p_st, h_short, qr_short).jaxpr
    self.assertFalse(_has_exact_multiset(_jaxpr_shapes(stream_jp), (batch, short_seq, short_n)))
    self.assertFalse(
        _has_exact_multiset(
            _compiled_hlo_shapes(run_streaming_compressor, p_st, h_short, qr_short), (batch, short_seq, short_n)
        )
    )

  def test_chunked_indexer_parity_forward_topk_ties_packed_sharded_and_kl_grads(self):
    # Force non-dividing multi-step scans across query (20 / 7), window (5 / 2), and head (4 / 3) axes.
    chunked_cfg = dataclasses.replace(
        self.cfg, index_query_chunk_size=7, index_window_chunk_size=2, index_head_chunk_size=3
    )
    unchunked_cfg = dataclasses.replace(
        self.cfg, index_query_chunk_size=1024, index_window_chunk_size=1024, index_head_chunk_size=1024
    )

    comp_chunk = _Component("attn_csa", _attention_module(chunked_cfg, "csa"), chunked_cfg)
    self.assert_parity(comp_chunk, lambda p, inp: {"out": comp_chunk.bind(p)(inp["hidden"])[0]})

    # Compare scores, streaming select_topk (including exact ties and packed segment_ids), and selected masks.
    comp_ref = _Component("attn_csa", _attention_module(unchunked_cfg, "csa"), unchunked_cfg)
    hidden = comp_ref.inputs()["hidden"]
    seg_packed = jnp.asarray(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        ],
        dtype=jnp.int32,
    )

    for seg in (None, seg_packed):
      docs = dsv4._documents(seg, *hidden.shape[:2])  # pylint: disable=protected-access
      win = dsv4._windows(docs, self.cfg.compress_rate_csa)  # pylint: disable=protected-access
      m_ref = comp_ref.bind(comp_ref.params)
      m_chk = comp_chunk.bind(comp_chunk.params)
      qr = m_ref.q_a_norm(m_ref.q_a_proj(hidden))
      scores_ref = m_ref.compressor.indexer.scores(hidden, qr, docs, win)
      scores_chk = m_chk.compressor.indexer.scores(hidden, qr, docs, win)
      np.testing.assert_allclose(np.asarray(scores_chk), np.asarray(scores_ref), rtol=1e-6, atol=1e-6)

      vis = dsv4._visible_windows(docs, win)  # pylint: disable=protected-access
      topk_ref, sel_ref = m_ref.compressor.indexer.select(scores_ref, vis)
      topk_stream = m_chk.compressor.indexer.select_topk(hidden, qr, docs, win)
      _, sel_chk, diag_chk = m_chk.compressor(hidden, qr, docs)
      np.testing.assert_array_equal(np.asarray(topk_stream), np.asarray(topk_ref))
      np.testing.assert_array_equal(np.asarray(diag_chk["topk_indices"]), np.asarray(topk_ref))
      np.testing.assert_array_equal(np.asarray(sel_chk), np.asarray(sel_ref))

    # Exact-tie test across window chunks: zeroing q_b_proj gives identical 0.0 scores across all visible windows,
    # verifying lowest-window-index tie-breaking matches unchunked jax.lax.top_k bit for bit.
    tie_params = jax.tree.map(lambda x: x, comp_chunk.params)
    tie_params["compressor"]["indexer"]["q_b_proj"]["kernel"] = jnp.zeros_like(
        tie_params["compressor"]["indexer"]["q_b_proj"]["kernel"]
    )
    docs_tie = dsv4._documents(None, *hidden.shape[:2])  # pylint: disable=protected-access
    win_tie = dsv4._windows(docs_tie, self.cfg.compress_rate_csa)  # pylint: disable=protected-access
    qr_tie = comp_ref.bind(tie_params).q_a_norm(comp_ref.bind(tie_params).q_a_proj(hidden))
    scores_tie = comp_ref.bind(tie_params).compressor.indexer.scores(hidden, qr_tie, docs_tie, win_tie)
    topk_tie_ref, _ = comp_ref.bind(tie_params).compressor.indexer.select(
        scores_tie, dsv4._visible_windows(docs_tie, win_tie)  # pylint: disable=protected-access
    )
    topk_tie_stream = comp_chunk.bind(tie_params).compressor.indexer.select_topk(hidden, qr_tie, docs_tie, win_tie)
    np.testing.assert_array_equal(np.asarray(topk_tie_stream), np.asarray(topk_tie_ref))

    # Opt-in csa_indexer_kl_loss forward + all-parameter gradients under jax.jit(jax.value_and_grad).
    loss_ref_cfg = dataclasses.replace(unchunked_cfg, indexer_loss_scaling_factor=0.5)
    loss_chk_cfg = dataclasses.replace(chunked_cfg, indexer_loss_scaling_factor=0.5)
    loss_ref = _Component("attn_csa", _attention_module(loss_ref_cfg, "csa"), loss_ref_cfg)
    loss_chk = _Component("attn_csa", _attention_module(loss_chk_cfg, "csa"), loss_chk_cfg)

    for seg in (None, seg_packed):
      val_ref, grads_ref = jax.jit(jax.value_and_grad(lambda p, x, s=seg: loss_ref.bind(p)(x, s)[1]["csa_indexer_loss"]))(
          loss_ref.params, hidden
      )
      val_chk, grads_chk = jax.jit(jax.value_and_grad(lambda p, x, s=seg: loss_chk.bind(p)(x, s)[1]["csa_indexer_loss"]))(
          loss_chk.params, hidden
      )
      np.testing.assert_allclose(float(val_chk), float(val_ref), rtol=1e-6, atol=1e-6)
      for (path, g_chk), (_, g_ref) in zip(
          jax.tree_util.tree_flatten_with_path(grads_chk)[0], jax.tree_util.tree_flatten_with_path(grads_ref)[0]
      ):
        ok, metrics = layer_parity.compare(np.asarray(g_chk), np.asarray(g_ref), _TOL)
        self.assertTrue(ok, f"KL grad mismatch at {path}: {metrics}")

    # 4-device sharded execution (Auto TP=2 where index_heads is sharded on 'tensor', plus Explicit reshard check).
    self.assertGreaterEqual(jax.device_count(), 4, "set XLA_FLAGS=--xla_force_host_platform_device_count=4")
    with nnx.use_eager_sharding(True):
      auto_mesh = jax.make_mesh(
          (1, 2, 2, 1),
          ("data", "fsdp", "tensor", "expert"),
          axis_types=(jax.sharding.AxisType.Auto,) * 4,
          devices=jax.devices()[:4],
      )
      with jax.sharding.set_mesh(auto_mesh):
        sh_comp = _Component("attn_csa", _attention_module(chunked_cfg, "csa", dsv4.DeepseekV4Sharding()), chunked_cfg)
        out_sh = nnx.jit(lambda m, h: m(h)[0])(sh_comp.bind(sh_comp.params), sh_comp.inputs()["hidden"])
        ok, metrics = layer_parity.compare(np.asarray(out_sh), GOLDEN["attn_csa/out/out"], _TOL)
        self.assertTrue(ok, f"sharded chunked CSA: {metrics}")

      # Also verify Explicit mesh mode on tensor-sharded head chunks (`index_heads -> 'tensor'`) does not raise
      # ShardingTypeError when slicing/reshaping heads inside `_score_tile`.
      exp_mesh = jax.make_mesh(
          (1, 2, 2, 1),
          ("data", "fsdp", "tensor", "expert"),
          axis_types=(jax.sharding.AxisType.Explicit,) * 4,
          devices=jax.devices()[:4],
      )
      with jax.sharding.set_mesh(exp_mesh):
        P = jax.sharding.PartitionSpec
        q_exp = jax.device_put(
            jnp.ones((2, 8, 4, 16), dtype=jnp.float32),
            jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, "tensor", None)),
        )
        w_exp = jax.device_put(
            jnp.ones((2, 8, 4), dtype=jnp.float32), jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, "tensor"))
        )
        k_exp = jax.device_put(
            jnp.ones((2, 4, 16), dtype=jnp.float32), jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, None))
        )
        tile_out = jax.jit(lambda q_i, w_i, k_i: dsv4._score_tile(q_i, w_i, k_i, head_scale=0.25, head_chunk_size=2))(  # pylint: disable=protected-access
            q_exp, w_exp, k_exp
        )
        np.testing.assert_allclose(np.asarray(tile_out), np.full((2, 8, 4), 16.0, dtype=np.float32), rtol=1e-6)

    if jax.default_backend() == "cpu":
      with jax.enable_x64(True):
        f64_chk_cfg = dataclasses.replace(
            _f64_config(), index_query_chunk_size=5, index_window_chunk_size=3, index_head_chunk_size=1
        )
        f64_comp = _Component("f64_attn_csa", _attention_module(f64_chk_cfg, "csa"), f64_chk_cfg)
        self.assert_parity(f64_comp, lambda p, inp: {"out": f64_comp.bind(p)(inp["hidden"])[0]}, tol=_FP64_TOL)

    # Verify outer scan_layers + jax.checkpoint composition (forward + backward) never crashes or hoists 4D scores.
    for scan_comp in (comp_chunk, loss_chk):
      stacked_params = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf], axis=0), scan_comp.params)

      def scanned_loss(sp, x, c=scan_comp):
        @jax.checkpoint
        def layer_step(carry, layer_p):
          y, aux = c.bind(layer_p)(carry)
          aux_sum = sum(aux.values(), jnp.zeros((), dtype=jnp.float32))
          return carry + 0.1 * y, aux_sum

        final_h, aux_vec = jax.lax.scan(layer_step, x, sp)
        return jnp.mean(final_h**2) + jnp.sum(aux_vec)

      s_val, (sp_grads, sx_grad) = jax.jit(jax.value_and_grad(scanned_loss, argnums=(0, 1)))(stacked_params, hidden)
      self.assertTrue(np.isfinite(float(s_val)))
      self.assertTrue(bool(jnp.all(jnp.isfinite(sx_grad))))
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(sp_grads)))

  def test_attention_and_kl_loss_never_materialize_dense_ss_or_bhsn_tensors(self):
    batch, seq, heads, idx_heads, idx_hd = 2, 320, 8, 16, 16
    n_csa, n_hca = seq // 4, seq // 8  # 80 CSA windows, 40 HCA windows; win_width = 64 + 16 - 1 = 79 << 320.
    hf = dict(
        MANIFEST["hf_config"],
        num_attention_heads=heads,
        sliding_window=16,
        index_n_heads=idx_heads,
        index_head_dim=idx_hd,
        index_topk=8,
    )
    q_probe = jnp.ones((batch, seq, heads, 16), dtype=jnp.float32)
    k_probe = jnp.ones((batch, seq, 16), dtype=jnp.float32)

    def unchunked_attn_probe(q_p, k_p):
      m_ss = jnp.einsum("bsd,btd->bst", q_p[:, :, 0], k_p) > -1.0
      logits_bhss = jnp.where(m_ss[:, None], jnp.einsum("bshd,btd->bhst", q_p, k_p), -1e30)
      return jnp.sum(jax.nn.softmax(logits_bhss, axis=-1))

    probe_jp = _jaxpr_shapes(jax.make_jaxpr(unchunked_attn_probe)(q_probe, k_probe).jaxpr)
    probe_hlo = _compiled_hlo_shapes(unchunked_attn_probe, q_probe, k_probe)
    for bad_probe in ((batch, seq, seq), (batch, heads, seq, seq)):
      self.assertTrue(_has_exact_multiset(probe_jp, bad_probe))
      self.assertTrue(_has_exact_multiset(probe_hlo, bad_probe))

    for kind, loss_scale in (("sliding", 0.0), ("hca", 0.0), ("csa", 0.0), ("csa", 0.5)):
      with self.subTest(kind=kind, loss_scale=loss_scale):
        cfg = dsv4.DeepseekV4Config.from_hf_config(
            hf,
            dtype="float32",
            weight_dtype="float32",
            indexer_loss_scaling_factor=loss_scale,
            attn_query_chunk_size=64,
            index_query_chunk_size=64,
            index_window_chunk_size=16,
            index_head_chunk_size=8,
        )
        attn = _attention_module(cfg, kind)
        gd, st = nnx.split(attn)
        pure_p = nnx.to_pure_dict(st)
        x_in = jnp.ones((batch, seq, cfg.hidden_size), dtype=jnp.float32) * 0.1

        def run_fn(p, x, g=gd, s=st):
          st_copy = jax.tree.map(lambda leaf: leaf, s)
          nnx.replace_by_pure_dict(st_copy, p)
          out, aux = nnx.merge(g, st_copy)(x)
          return jnp.sum(out) + sum(aux.values(), jnp.zeros((), dtype=jnp.float32))

        forbidden = [
            (batch, seq, seq),
            (batch, heads, seq, seq),
            (batch, heads, seq, n_csa),
            (batch, heads, seq, n_hca),
            (batch, seq, n_csa),
            (batch, seq, n_hca),
        ]
        for label, fn_to_check in (("fwd", run_fn), ("bwd", jax.value_and_grad(run_fn, argnums=(0, 1)))):
          jp_shapes = _jaxpr_shapes(jax.make_jaxpr(fn_to_check)(pure_p, x_in).jaxpr)
          hlo_shapes = _compiled_hlo_shapes(fn_to_check, pure_p, x_in)
          for shape in forbidden:
            self.assertFalse(
                _has_exact_multiset(jp_shapes, shape),
                f"{kind} (loss={loss_scale}) {label} jaxpr materialized forbidden {shape}",
            )
            self.assertFalse(
                _has_exact_multiset(hlo_shapes, shape),
                f"{kind} (loss={loss_scale}) {label} compiled HLO materialized forbidden {shape}",
            )

    # Direct standalone csa_indexer_kl_loss memory bound + numerical parity check across query & window tiles.
    rng = np.random.default_rng(7)
    s_in = jnp.asarray(rng.standard_normal((batch, seq, n_csa), dtype=np.float32))
    q_in = jnp.asarray(rng.standard_normal((batch, seq, heads, 32), dtype=np.float32))
    k_in = jnp.asarray(rng.standard_normal((batch, n_csa, 32), dtype=np.float32))
    vis_in = (jnp.arange(n_csa)[None, None, :] < ((jnp.arange(seq)[None, :, None] + 1) // 4)) & jnp.ones(
        (batch, 1, 1), dtype=bool
    )

    def kl_chunked(scores_arg):
      return dsv4.csa_indexer_kl_loss(
          scores_arg, q_in, k_in, vis_in, scale=0.25, query_chunk_size=64, window_chunk_size=16
      )

    for label, fn_kl in (("fwd", kl_chunked), ("bwd", jax.value_and_grad(kl_chunked))):
      jp_s = _jaxpr_shapes(jax.make_jaxpr(fn_kl)(s_in).jaxpr)
      hlo_s = _compiled_hlo_shapes(fn_kl, s_in)
      for bad in ((batch, heads, seq, n_csa), (batch, heads, 64, n_csa)):
        self.assertFalse(_has_exact_multiset(jp_s, bad), f"standalone KL {label} jaxpr materialized {bad}")
        self.assertFalse(_has_exact_multiset(hlo_s, bad), f"standalone KL {label} HLO materialized {bad}")

  def test_chunked_attention_parity_packed_bf16_f64_and_explicit_mesh(self):
    # Standalone csa_indexer_kl_loss multi-tile query & window parity (forward + gradient).
    rng = np.random.default_rng(7)
    s_in = jnp.asarray(rng.standard_normal((2, 192, 48), dtype=np.float32))
    q_in = jnp.asarray(rng.standard_normal((2, 192, 8, 32), dtype=np.float32))
    k_in = jnp.asarray(rng.standard_normal((2, 48, 32), dtype=np.float32))
    vis_in = (jnp.arange(48)[None, None, :] < ((jnp.arange(192)[None, :, None] + 1) // 4)) & jnp.ones(
        (2, 1, 1), dtype=bool
    )
    val_chk, grad_chk = jax.jit(
        jax.value_and_grad(
            lambda sc: dsv4.csa_indexer_kl_loss(
                sc, q_in, k_in, vis_in, scale=0.25, query_chunk_size=64, window_chunk_size=16
            )
        )
    )(s_in)
    val_ref, grad_ref = jax.jit(
        jax.value_and_grad(
            lambda sc: dsv4.csa_indexer_kl_loss(
                sc, q_in, k_in, vis_in, scale=0.25, query_chunk_size=1024, window_chunk_size=1024
            )
        )
    )(s_in)
    np.testing.assert_allclose(float(val_chk), float(val_ref), rtol=1e-6, atol=1e-6)
    ok_g, metrics_g = layer_parity.compare(np.asarray(grad_chk), np.asarray(grad_ref), _TOL)
    self.assertTrue(ok_g, f"standalone chunked KL grad mismatch: {metrics_g}")

    # Non-dividing query chunk (seq=20, chunk=7 -> 3 chunks with tail pad) and multi-chunk windows.
    chk_cfg = dataclasses.replace(
        self.cfg, attn_query_chunk_size=7, index_query_chunk_size=7, index_window_chunk_size=2, index_head_chunk_size=3
    )
    ref_cfg = dataclasses.replace(
        self.cfg,
        attn_query_chunk_size=1024,
        index_query_chunk_size=1024,
        index_window_chunk_size=1024,
        index_head_chunk_size=1024,
    )
    seg_packed = jnp.asarray(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        ],
        dtype=jnp.int32,
    )
    for kind in ("sliding", "hca", "csa"):
      with self.subTest(kind=kind):
        comp_chk = _Component(f"attn_{kind}", _attention_module(chk_cfg, kind), chk_cfg)
        self.assert_parity(comp_chk, lambda p, inp, c=comp_chk: {"out": c.bind(p)(inp["hidden"])[0]})
        comp_ref = _Component(f"attn_{kind}", _attention_module(ref_cfg, kind), ref_cfg)
        hidden = comp_ref.inputs()["hidden"]
        for seg in (None, seg_packed):
          y_ref = jax.jit(lambda p, x, s=seg, c=comp_ref: c.bind(p)(x, s)[0])(comp_ref.params, hidden)
          y_chk = jax.jit(lambda p, x, s=seg, c=comp_chk: c.bind(p)(x, s)[0])(comp_chk.params, hidden)
          ok_y, metrics_y = layer_parity.compare(np.asarray(y_chk), np.asarray(y_ref), _TOL)
          self.assertTrue(ok_y, f"{kind} seg={seg is not None} out: {metrics_y}")
          _, g_ref = jax.jit(jax.value_and_grad(lambda p, x, s=seg, c=comp_ref: jnp.sum(c.bind(p)(x, s)[0])))(
              comp_ref.params, hidden
          )
          _, g_chk = jax.jit(jax.value_and_grad(lambda p, x, s=seg, c=comp_chk: jnp.sum(c.bind(p)(x, s)[0])))(
              comp_chk.params, hidden
          )
          for (pth, gc), (_, gr) in zip(
              jax.tree_util.tree_flatten_with_path(g_chk)[0], jax.tree_util.tree_flatten_with_path(g_ref)[0]
          ):
            ok, metrics = layer_parity.compare(np.asarray(gc), np.asarray(gr), _TOL)
            self.assertTrue(ok, f"{kind} seg={seg is not None} grad {pth}: {metrics}")

    # Exercise HCA multi-window-chunk 2-pass online softmax in bfloat16 (verifying exact probs.astype(bf16) rounding).
    hf_hca_bf16 = dict(
        MANIFEST["hf_config"], compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4}
    )
    bf16_chk = dsv4.DeepseekV4Config.from_hf_config(
        hf_hca_bf16, dtype="bfloat16", weight_dtype="bfloat16", attn_query_chunk_size=7, index_window_chunk_size=2
    )
    bf16_ref = dataclasses.replace(bf16_chk, attn_query_chunk_size=1024, index_window_chunk_size=1024)
    m_bf16_chk = _attention_module(bf16_chk, "hca")
    _, st_bf16 = nnx.split(m_bf16_chk)
    p_bf16 = nnx.to_pure_dict(st_bf16)
    gd_bf16_chk, _ = nnx.split(m_bf16_chk)
    gd_bf16_ref, st_bf16_ref = nnx.split(_attention_module(bf16_ref, "hca"))
    x_bf16 = jnp.asarray(
        np.random.default_rng(11).standard_normal((2, 23, bf16_chk.hidden_size), dtype=np.float32), dtype=jnp.bfloat16
    )

    def run_bf16(gd_i, st_i, p_i, x_i):
      st_c = jax.tree.map(lambda leaf: leaf, st_i)
      nnx.replace_by_pure_dict(st_c, p_i)
      return nnx.merge(gd_i, st_c)(x_i, None)[0]

    y_bf16_chk = jax.jit(lambda p_i, x_i: run_bf16(gd_bf16_chk, st_bf16, p_i, x_i))(p_bf16, x_bf16)
    y_bf16_ref = jax.jit(lambda p_i, x_i: run_bf16(gd_bf16_ref, st_bf16_ref, p_i, x_i))(p_bf16, x_bf16)
    np.testing.assert_allclose(
        np.asarray(y_bf16_chk, dtype=np.float32), np.asarray(y_bf16_ref, dtype=np.float32), rtol=1e-4, atol=1e-4
    )

    if jax.default_backend() == "cpu":
      with jax.enable_x64(True):
        f64_chk = dataclasses.replace(
            _f64_config(), attn_query_chunk_size=5, index_query_chunk_size=5, index_window_chunk_size=1
        )
        for kind in ("sliding", "hca", "csa"):
          f64_c = _Component(f"f64_attn_{kind}", _attention_module(f64_chk, kind), f64_chk)
          self.assert_parity(f64_c, lambda p, inp, c=f64_c: {"out": c.bind(p)(inp["hidden"])[0]}, tol=_FP64_TOL)

    # Full forward + backward under 4-device Auto mesh (fsdp=2, tensor=2) across all 3 layer types + KL loss, plus
    # Explicit-mesh sharding check on `_attend_chunks` and `csa_indexer_kl_loss` (`q` on `('fsdp', None, 'tensor', None)`).
    with nnx.use_eager_sharding(True):
      auto_mesh = jax.make_mesh(
          (1, 2, 2, 1),
          ("data", "fsdp", "tensor", "expert"),
          axis_types=(jax.sharding.AxisType.Auto,) * 4,
          devices=jax.devices()[:4],
      )
      with jax.sharding.set_mesh(auto_mesh):
        shd = dsv4.DeepseekV4Sharding()
        for kind, lscale in (("sliding", 0.0), ("hca", 0.0), ("csa", 0.0), ("csa", 0.5)):
          acfg = dataclasses.replace(chk_cfg, indexer_loss_scaling_factor=lscale)
          aattn = _attention_module(acfg, kind, shd)
          agd, ast_state = nnx.split(aattn)
          ap = nnx.to_pure_dict(ast_state)
          ax = jnp.ones((2, 20, acfg.hidden_size), dtype=jnp.float32) * 0.1

          def auto_loss(p_in, x_in, g=agd, s=ast_state):
            st_c = jax.tree.map(lambda leaf: leaf, s)
            nnx.replace_by_pure_dict(st_c, p_in)
            out_y, aux_y = nnx.merge(g, st_c)(x_in)
            return jnp.sum(out_y) + sum(aux_y.values(), jnp.zeros((), dtype=jnp.float32))

          av, ag = jax.jit(jax.value_and_grad(auto_loss))(ap, ax)
          self.assertTrue(np.isfinite(float(av)))
          self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(ag)))

      exp_mesh = jax.make_mesh(
          (1, 2, 2, 1),
          ("data", "fsdp", "tensor", "expert"),
          axis_types=(jax.sharding.AxisType.Explicit,) * 4,
          devices=jax.devices()[:4],
      )
      with jax.sharding.set_mesh(exp_mesh):
        P = jax.sharding.PartitionSpec
        q_exp = jax.device_put(
            jnp.ones((2, 20, 4, 16), dtype=jnp.float32) * 0.2,
            jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, "tensor", None)),
        )
        kv_exp = jax.device_put(
            jnp.ones((2, 20, 16), dtype=jnp.float32) * 0.2,
            jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, None)),
        )
        comp_exp = jax.device_put(
            jnp.ones((2, 5, 16), dtype=jnp.float32) * 0.2,
            jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, None)),
        )
        sinks_exp = jax.device_put(jnp.zeros((4,), dtype=jnp.float32), jax.sharding.NamedSharding(exp_mesh, P("tensor")))
        scores_exp = jax.device_put(
            jnp.ones((2, 20, 5), dtype=jnp.float32) * 0.1,
            jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, None)),
        )
        topk_exp = jax.device_put(
            jnp.broadcast_to(jnp.asarray([0, -1], dtype=jnp.int32), (2, 20, 2)),
            jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, None)),
        )

        def exp_kernels(q_i, kv_i, comp_i, sk_i, sc_i):
          docs_e = dsv4._documents(None, 2, 20)  # pylint: disable=protected-access
          win_e = dsv4._windows(docs_e, 4)  # pylint: disable=protected-access
          vis_e = dsv4._visible_windows(docs_e, win_e)  # pylint: disable=protected-access
          y_csa = dsv4._attend_chunks(  # pylint: disable=protected-access
              q_i,
              kv_i,
              sk_i,
              docs_e,
              sliding_window=4,
              scale=0.25,
              acc=jnp.float32,
              dot_prec=jax.lax.Precision.HIGHEST,
              query_chunk_size=7,
              window_chunk_size=2,
              compressed=comp_i,
              topk_indices=topk_exp,
          )
          y_hca = dsv4._attend_chunks(  # pylint: disable=protected-access
              q_i,
              kv_i,
              sk_i,
              docs_e,
              sliding_window=4,
              scale=0.25,
              acc=jnp.float32,
              dot_prec=jax.lax.Precision.HIGHEST,
              query_chunk_size=7,
              window_chunk_size=2,
              compressed=comp_i,
              hca_win=win_e,
          )
          kl_e = dsv4.csa_indexer_kl_loss(sc_i, q_i, comp_i, vis_e, scale=0.25, query_chunk_size=7, window_chunk_size=2)
          return jnp.sum(jax.sharding.reshard(y_csa, P())) + jnp.sum(jax.sharding.reshard(y_hca, P())) + kl_e

        ev, eg = jax.jit(jax.value_and_grad(exp_kernels, argnums=(0, 1, 2, 3, 4)))(
            q_exp, kv_exp, comp_exp, sinks_exp, scores_exp
        )
        self.assertTrue(np.isfinite(float(ev)))
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in eg))

  def test_attention_emits_remat_checkpoint_names(self):
    def collect_names(jp: jax.extend.core.Jaxpr) -> set[str]:
      found: set[str] = set()
      seen: set[int] = set()

      def visit(sub: jax.extend.core.Jaxpr) -> None:
        if id(sub) in seen:
          return
        seen.add(id(sub))
        for eqn in sub.eqns:
          if eqn.primitive.name == "name" and "name" in eqn.params:
            found.add(str(eqn.params["name"]))
          for val in eqn.params.values():
            if isinstance(val, jax.extend.core.ClosedJaxpr):
              visit(val.jaxpr)
            elif isinstance(val, jax.extend.core.Jaxpr):
              visit(val)
            elif isinstance(val, (tuple, list)):
              for item in val:
                if isinstance(item, jax.extend.core.ClosedJaxpr):
                  visit(item.jaxpr)
                elif isinstance(item, jax.extend.core.Jaxpr):
                  visit(item)

      visit(jp)
      return found

    expected = {"query_proj", "kv_proj", "context", "out_proj"}
    policy = jax.checkpoint_policies.save_only_these_names(*expected)
    for kind in ("sliding", "hca", "csa"):
      comp = _Component(f"attn_{kind}", _attention_module(self.cfg, kind), self.cfg)
      hidden = comp.inputs()["hidden"]
      jp = jax.make_jaxpr(lambda p, x, c=comp: c.bind(p)(x)[0])(comp.params, hidden).jaxpr
      self.assertTrue(expected.issubset(collect_names(jp)), f"{kind} checkpoint names: {collect_names(jp)}")

      @functools.partial(jax.checkpoint, policy=policy, prevent_cse=False)
      def rematted(p, x, c=comp):
        return c.bind(p)(x)[0]

      y_remat = jax.jit(rematted)(comp.params, hidden)
      y_plain = jax.jit(lambda p, x, c=comp: c.bind(p)(x)[0])(comp.params, hidden)
      ok_y, metrics_y = layer_parity.compare(np.asarray(y_remat), np.asarray(y_plain), _TOL)
      self.assertTrue(ok_y, f"{kind} remat out: {metrics_y}")
      _, g_remat = jax.jit(jax.value_and_grad(lambda p, x, fn=rematted: jnp.sum(fn(p, x))))(comp.params, hidden)
      _, g_plain = jax.jit(jax.value_and_grad(lambda p, x, c=comp: jnp.sum(c.bind(p)(x)[0])))(comp.params, hidden)
      for (pth, gr), (_, gp) in zip(
          jax.tree_util.tree_flatten_with_path(g_remat)[0], jax.tree_util.tree_flatten_with_path(g_plain)[0]
      ):
        ok, metrics = layer_parity.compare(np.asarray(gr), np.asarray(gp), _TOL)
        self.assertTrue(ok, f"{kind} remat grad {pth}: {metrics}")


if __name__ == "__main__":
  absltest.main()
