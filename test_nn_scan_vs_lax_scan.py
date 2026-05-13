"""
CHALLENGER-4: nn.scan vs jax.lax.scan jaxpr/HLO/memory comparison.

Tests whether nn.scan(variable_broadcast, variable_axes) produces different
jaxpr structure than raw jax.lax.scan with closure capture + manual stacking.

Key insight from docs: nn.scan __call__ must be (self, carry, *xs) -> (carry, ys).
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from flax import linen as nn
import functools

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH = 2
SEQ = 128
D_MODEL = 256
D_FF = 512
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
NUM_LAYERS = 4  # scan length
DTYPE = jnp.float32

print(
    f"Config: batch={BATCH}, seq={SEQ}, d_model={D_MODEL}, d_ff={D_FF}, "
    f"num_heads={NUM_HEADS}, head_dim={HEAD_DIM}, layers={NUM_LAYERS}"
)


# ===================================================================
# Shared computation: transformer-like block
# ===================================================================


def transformer_block_fn(x, wq, wk, wv, wo, w1, w2):
  """Pure function: one transformer-like layer."""
  head_dim = HEAD_DIM

  # Attention
  q = jnp.einsum("bsd,dhk->bshk", x, wq)
  k = jnp.einsum("bsd,dhk->bshk", x, wk)
  v = jnp.einsum("bsd,dhk->bshk", x, wv)
  attn = jnp.einsum("bshk,bthk->bhst", q, k)
  attn = jax.nn.softmax(attn / jnp.sqrt(float(head_dim)), axis=-1)
  attn_out = jnp.einsum("bhst,bthk->bshk", attn, v)
  attn_out = jnp.einsum("bshk,hkd->bsd", attn_out, wo)

  # FF
  residual = x + attn_out
  h = jnp.dot(residual, w1)
  h = jax.nn.gelu(h)
  ff_out = jnp.dot(h, w2)
  return residual + ff_out


# ===================================================================
# PART A: nn.scan with variable_axes (Linen pattern)
# ===================================================================


class TransformerScanBlock(nn.Module):
  """Block with correct nn.scan signature: (carry, xs) -> (carry, ys)."""

  d_model: int = D_MODEL
  d_ff: int = D_FF
  num_heads: int = NUM_HEADS

  @nn.compact
  def __call__(self, carry, _):
    """carry=x, xs=None (dummy scan input). Returns (new_carry, None)."""
    x = carry
    head_dim = self.d_model // self.num_heads

    wq = self.param("wq", nn.initializers.lecun_normal(), (self.d_model, self.num_heads, head_dim))
    wk = self.param("wk", nn.initializers.lecun_normal(), (self.d_model, self.num_heads, head_dim))
    wv = self.param("wv", nn.initializers.lecun_normal(), (self.d_model, self.num_heads, head_dim))
    wo = self.param("wo", nn.initializers.lecun_normal(), (self.num_heads, head_dim, self.d_model))
    w1 = self.param("w1", nn.initializers.lecun_normal(), (self.d_model, self.d_ff))
    w2 = self.param("w2", nn.initializers.lecun_normal(), (self.d_ff, self.d_model))

    out = transformer_block_fn(x, wq, wk, wv, wo, w1, w2)
    return out, None


class NNScanAxesPipeline(nn.Module):
  """nn.scan with variable_axes={'params': 0}: params stacked along axis 0."""

  num_layers: int = NUM_LAYERS

  @nn.compact
  def __call__(self, x):
    ScanLayer = nn.scan(
        nn.remat(TransformerScanBlock),
        variable_axes={"params": 0},
        variable_broadcast=False,
        split_rngs={"params": True},
        length=self.num_layers,
    )
    y, _ = ScanLayer(name="scanned")(x, None)
    return y


class NNScanBroadcastPipeline(nn.Module):
  """nn.scan with variable_broadcast='params': params shared (constants)."""

  num_layers: int = NUM_LAYERS

  @nn.compact
  def __call__(self, x):
    ScanLayer = nn.scan(
        nn.remat(TransformerScanBlock),
        variable_broadcast="params",
        split_rngs={"params": False},
        length=self.num_layers,
    )
    y, _ = ScanLayer(name="scanned")(x, None)
    return y


def analyze_model(name, model, rng, x):
  """Analyze a Linen model: init, jaxpr, HLO."""
  print(f"\n{'=' * 70}")
  print(f"  {name}")
  print(f"{'=' * 70}")

  variables = model.init(rng, x)
  param_leaves = jax.tree.leaves(variables["params"])
  total_params = sum(p.size for p in param_leaves)
  print(f"  Total params: {total_params:,}")
  print(f"  Num param arrays: {len(param_leaves)}")
  print(f"  Param shapes (first 4): {[p.shape for p in param_leaves[:4]]}")

  def loss_fn(params, x_in):
    return jnp.sum(model.apply({"params": params}, x_in))

  # Forward jaxpr
  fwd_jaxpr = jax.make_jaxpr(lambda p, xi: model.apply({"params": p}, xi))(variables["params"], x)
  fwd_eqns = fwd_jaxpr.jaxpr.eqns
  print(f"  Forward jaxpr equations: {len(fwd_eqns)}")
  scan_eqns = [e for e in fwd_eqns if e.primitive.name == "scan"]
  print(f"  Forward scan primitives: {len(scan_eqns)}")
  for i, se in enumerate(scan_eqns):
    nc = se.params.get("num_consts", "?")
    ncarry = se.params.get("num_carry", "?")
    length = se.params.get("length", "?")
    inner = se.params.get("jaxpr", None)
    inner_count = len(inner.jaxpr.eqns) if inner else "?"
    print(f"    scan[{i}]: consts={nc}, carry={ncarry}, length={length}, " f"inner_eqns={inner_count}")

  # Grad jaxpr
  grad_jaxpr = jax.make_jaxpr(jax.grad(loss_fn))(variables["params"], x)
  grad_eqns = grad_jaxpr.jaxpr.eqns
  print(f"  Grad jaxpr equations: {len(grad_eqns)}")
  grad_scans = [e for e in grad_eqns if e.primitive.name == "scan"]
  print(f"  Grad scan primitives: {len(grad_scans)}")
  for i, se in enumerate(grad_scans):
    nc = se.params.get("num_consts", "?")
    ncarry = se.params.get("num_carry", "?")
    length = se.params.get("length", "?")
    inner = se.params.get("jaxpr", None)
    inner_count = len(inner.jaxpr.eqns) if inner else "?"
    print(f"    grad_scan[{i}]: consts={nc}, carry={ncarry}, length={length}, " f"inner_eqns={inner_count}")

  # HLO
  compiled = jax.jit(jax.grad(loss_fn)).lower(variables["params"], x).compile()
  hlo_text = compiled.as_text()
  while_count = hlo_text.count("while(")
  hlo_size = len(hlo_text)
  print(f"  HLO while-loop count: {while_count}")
  print(f"  HLO size (chars): {hlo_size:,}")

  return {
      "params": variables["params"],
      "hlo": hlo_text,
      "fwd_jaxpr": fwd_jaxpr,
      "grad_jaxpr": grad_jaxpr,
      "fwd_eqns": len(fwd_eqns),
      "grad_eqns": len(grad_eqns),
      "fwd_scans": len(scan_eqns),
      "grad_scans": len(grad_scans),
      "while_loops": while_count,
      "hlo_size": hlo_size,
      "total_params": total_params,
  }


# ===================================================================
# PART B: jax.lax.scan with stacked params as xs
# ===================================================================


def make_layer_params(rng):
  """Create one layer's worth of params."""
  keys = random.split(rng, 6)
  return {
      "wq": random.normal(keys[0], (D_MODEL, NUM_HEADS, HEAD_DIM), dtype=DTYPE) * 0.02,
      "wk": random.normal(keys[1], (D_MODEL, NUM_HEADS, HEAD_DIM), dtype=DTYPE) * 0.02,
      "wv": random.normal(keys[2], (D_MODEL, NUM_HEADS, HEAD_DIM), dtype=DTYPE) * 0.02,
      "wo": random.normal(keys[3], (NUM_HEADS, HEAD_DIM, D_MODEL), dtype=DTYPE) * 0.02,
      "w1": random.normal(keys[4], (D_MODEL, D_FF), dtype=DTYPE) * 0.02,
      "w2": random.normal(keys[5], (D_FF, D_MODEL), dtype=DTYPE) * 0.02,
  }


def make_stacked_params(rng):
  """Stack params: each array has leading dim = num_layers."""
  keys = random.split(rng, NUM_LAYERS)
  layers = [make_layer_params(k) for k in keys]
  return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *layers)


def analyze_lax_scan(name, stacked_params, x, use_remat=True):
  """Analyze jax.lax.scan version."""
  print(f"\n{'=' * 70}")
  print(f"  {name}")
  print(f"{'=' * 70}")

  param_leaves = jax.tree.leaves(stacked_params)
  total_params = sum(p.size for p in param_leaves)
  print(f"  Total params: {total_params:,}")
  print(f"  Num param arrays: {len(param_leaves)}")
  print(f"  Param shapes (first 4): {[p.shape for p in param_leaves[:4]]}")

  def forward(sp, x_in):
    def scan_body(carry, lp):
      fn = jax.checkpoint(layer_fn_dict, prevent_cse=False) if use_remat else layer_fn_dict
      return fn(lp, carry), None

    final, _ = jax.lax.scan(scan_body, x_in, sp, length=NUM_LAYERS)
    return final

  def loss_fn(sp, x_in):
    return jnp.sum(forward(sp, x_in))

  # Forward jaxpr
  fwd_jaxpr = jax.make_jaxpr(forward)(stacked_params, x)
  fwd_eqns = fwd_jaxpr.jaxpr.eqns
  print(f"  Forward jaxpr equations: {len(fwd_eqns)}")
  scan_eqns = [e for e in fwd_eqns if e.primitive.name == "scan"]
  print(f"  Forward scan primitives: {len(scan_eqns)}")
  for i, se in enumerate(scan_eqns):
    nc = se.params.get("num_consts", "?")
    ncarry = se.params.get("num_carry", "?")
    length = se.params.get("length", "?")
    inner = se.params.get("jaxpr", None)
    inner_count = len(inner.jaxpr.eqns) if inner else "?"
    print(f"    scan[{i}]: consts={nc}, carry={ncarry}, length={length}, " f"inner_eqns={inner_count}")

  # Grad jaxpr
  grad_jaxpr = jax.make_jaxpr(jax.grad(loss_fn))(stacked_params, x)
  grad_eqns = grad_jaxpr.jaxpr.eqns
  print(f"  Grad jaxpr equations: {len(grad_eqns)}")
  grad_scans = [e for e in grad_eqns if e.primitive.name == "scan"]
  print(f"  Grad scan primitives: {len(grad_scans)}")
  for i, se in enumerate(grad_scans):
    nc = se.params.get("num_consts", "?")
    ncarry = se.params.get("num_carry", "?")
    length = se.params.get("length", "?")
    inner = se.params.get("jaxpr", None)
    inner_count = len(inner.jaxpr.eqns) if inner else "?"
    print(f"    grad_scan[{i}]: consts={nc}, carry={ncarry}, length={length}, " f"inner_eqns={inner_count}")

  # HLO
  compiled = jax.jit(jax.grad(loss_fn)).lower(stacked_params, x).compile()
  hlo_text = compiled.as_text()
  while_count = hlo_text.count("while(")
  hlo_size = len(hlo_text)
  print(f"  HLO while-loop count: {while_count}")
  print(f"  HLO size (chars): {hlo_size:,}")

  return {
      "hlo": hlo_text,
      "fwd_jaxpr": fwd_jaxpr,
      "grad_jaxpr": grad_jaxpr,
      "fwd_eqns": len(fwd_eqns),
      "grad_eqns": len(grad_eqns),
      "fwd_scans": len(scan_eqns),
      "grad_scans": len(grad_scans),
      "while_loops": while_count,
      "hlo_size": hlo_size,
      "total_params": total_params,
  }


def layer_fn_dict(params_i, x):
  """Layer forward from a dict of params."""
  return transformer_block_fn(
      x, params_i["wq"], params_i["wk"], params_i["wv"], params_i["wo"], params_i["w1"], params_i["w2"]
  )


# ===================================================================
# PART C: jax.lax.scan with CLOSURE capture (params as constants)
# ===================================================================


def analyze_lax_scan_closure(name, stacked_params, x, use_remat=True):
  """
  lax.scan where stacked params are closed over and indexed per iteration.
  Tests: closure-captured params become scan CONSTANTS.
  """
  print(f"\n{'=' * 70}")
  print(f"  {name}")
  print(f"{'=' * 70}")

  param_leaves = jax.tree.leaves(stacked_params)
  total_params = sum(p.size for p in param_leaves)
  print(f"  Total params: {total_params:,}")

  def forward(sp, x_in):
    def scan_body(carry, idx):
      # Dynamic index into closed-over stacked params
      lp = jax.tree.map(lambda arr: arr[idx], sp)
      fn = jax.checkpoint(layer_fn_dict, prevent_cse=False) if use_remat else layer_fn_dict
      return fn(lp, carry), None

    indices = jnp.arange(NUM_LAYERS)
    final, _ = jax.lax.scan(scan_body, x_in, indices, length=NUM_LAYERS)
    return final

  def loss_fn(sp, x_in):
    return jnp.sum(forward(sp, x_in))

  # Forward jaxpr
  fwd_jaxpr = jax.make_jaxpr(forward)(stacked_params, x)
  fwd_eqns = fwd_jaxpr.jaxpr.eqns
  print(f"  Forward jaxpr equations: {len(fwd_eqns)}")
  scan_eqns = [e for e in fwd_eqns if e.primitive.name == "scan"]
  print(f"  Forward scan primitives: {len(scan_eqns)}")
  for i, se in enumerate(scan_eqns):
    nc = se.params.get("num_consts", "?")
    ncarry = se.params.get("num_carry", "?")
    length = se.params.get("length", "?")
    inner = se.params.get("jaxpr", None)
    inner_count = len(inner.jaxpr.eqns) if inner else "?"
    print(f"    scan[{i}]: consts={nc}, carry={ncarry}, length={length}, " f"inner_eqns={inner_count}")

  # Grad jaxpr
  grad_jaxpr = jax.make_jaxpr(jax.grad(loss_fn))(stacked_params, x)
  grad_eqns = grad_jaxpr.jaxpr.eqns
  print(f"  Grad jaxpr equations: {len(grad_eqns)}")
  grad_scans = [e for e in grad_eqns if e.primitive.name == "scan"]
  print(f"  Grad scan primitives: {len(grad_scans)}")
  for i, se in enumerate(grad_scans):
    nc = se.params.get("num_consts", "?")
    ncarry = se.params.get("num_carry", "?")
    length = se.params.get("length", "?")
    inner = se.params.get("jaxpr", None)
    inner_count = len(inner.jaxpr.eqns) if inner else "?"
    print(f"    grad_scan[{i}]: consts={nc}, carry={ncarry}, length={length}, " f"inner_eqns={inner_count}")

  # HLO
  compiled = jax.jit(jax.grad(loss_fn)).lower(stacked_params, x).compile()
  hlo_text = compiled.as_text()
  while_count = hlo_text.count("while(")
  hlo_size = len(hlo_text)
  print(f"  HLO while-loop count: {while_count}")
  print(f"  HLO size (chars): {hlo_size:,}")

  return {
      "hlo": hlo_text,
      "fwd_jaxpr": fwd_jaxpr,
      "grad_jaxpr": grad_jaxpr,
      "fwd_eqns": len(fwd_eqns),
      "grad_eqns": len(grad_eqns),
      "fwd_scans": len(scan_eqns),
      "grad_scans": len(grad_scans),
      "while_loops": while_count,
      "hlo_size": hlo_size,
      "total_params": total_params,
  }


# ===================================================================
# Comparison
# ===================================================================


def compare_results(results_dict):
  """Side-by-side comparison."""
  print(f"\n{'=' * 70}")
  print("  SIDE-BY-SIDE COMPARISON")
  print(f"{'=' * 70}")

  metrics = ["total_params", "fwd_eqns", "grad_eqns", "fwd_scans", "grad_scans", "while_loops", "hlo_size"]

  names = list(results_dict.keys())
  header = f"  {'Metric':<25}" + "".join(f"{n:>22}" for n in names)
  print(header)
  print("  " + "-" * (25 + 22 * len(names)))

  for m in metrics:
    row = f"  {m:<25}"
    vals = []
    for n in names:
      v = results_dict[n].get(m, "N/A")
      vals.append(v)
      row += f"{v:>22,}" if isinstance(v, int) else f"{v!s:>22}"
    print(row)
    unique_vals = set(v for v in vals if v != "N/A")
    if len(unique_vals) > 1:
      print(f"  {'':25}  *** DIFFERENT ***")

  # Detailed scan structure comparison
  print(f"\n  SCAN STRUCTURE DETAILS (grad jaxpr):")
  for name, res in results_dict.items():
    grad_jaxpr = res["grad_jaxpr"]
    scans = [e for e in grad_jaxpr.jaxpr.eqns if e.primitive.name == "scan"]
    print(f"\n    {name}:")
    for i, se in enumerate(scans):
      nc = se.params.get("num_consts", "?")
      ncarry = se.params.get("num_carry", "?")
      length = se.params.get("length", "?")
      n_in = len(se.invars) if hasattr(se, "invars") else "?"
      n_out = len(se.outvars) if hasattr(se, "outvars") else "?"
      print(f"      scan[{i}]: consts={nc}, carry={ncarry}, length={length}, " f"in_vars={n_in}, out_vars={n_out}")
      inner = se.params.get("jaxpr", None)
      if inner:
        prim_counts = {}
        for eq in inner.jaxpr.eqns:
          pn = eq.primitive.name
          prim_counts[pn] = prim_counts.get(pn, 0) + 1
        top5 = sorted(prim_counts.items(), key=lambda x: -x[1])[:5]
        print(f"        inner top primitives: {top5}")

  # HLO keyword comparison
  print(f"\n  HLO KEYWORD COMPARISON:")
  keywords = ["while(", "dynamic-slice", "dynamic-update-slice", "scatter(", "gather(", "reduce(", "dot("]
  header = f"  {'keyword':<30}" + "".join(f"{n:>22}" for n in names)
  print(header)
  for kw in keywords:
    row = f"  {kw:<30}"
    for n in names:
      count = results_dict[n]["hlo"].count(kw)
      row += f"{count:>22}"
    print(row)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
  print("=" * 70)
  print("CHALLENGER-4: nn.scan vs jax.lax.scan structural comparison")
  print("=" * 70)
  print(f"JAX version: {jax.__version__}")
  print(f"Devices: {jax.devices()}")

  rng = random.PRNGKey(0)
  x = jnp.ones((BATCH, SEQ, D_MODEL), dtype=DTYPE)

  results = {}

  # A: nn.scan with variable_axes (Linen standard pattern)
  model_a = NNScanAxesPipeline()
  results["nn.scan(axes)"] = analyze_model("A: nn.scan variable_axes={'params':0}", model_a, rng, x)

  # B: nn.scan with variable_broadcast (shared-weight pattern)
  model_b = NNScanBroadcastPipeline()
  results["nn.scan(bcast)"] = analyze_model("B: nn.scan variable_broadcast='params'", model_b, rng, x)

  # C: jax.lax.scan with stacked params as xs
  stacked_params = make_stacked_params(random.PRNGKey(42))
  results["lax.scan(xs)"] = analyze_lax_scan("C: lax.scan stacked params as xs", stacked_params, x)

  # D: jax.lax.scan with closure capture
  results["lax.scan(closure)"] = analyze_lax_scan_closure("D: lax.scan closure + dynamic index", stacked_params, x)

  # E: jax.lax.scan without remat
  results["lax.scan(no_rem)"] = analyze_lax_scan("E: lax.scan NO remat (baseline)", stacked_params, x, use_remat=False)

  compare_results(results)

  # Final verdict
  print(f"\n{'=' * 70}")
  print("VERDICT")
  print(f"{'=' * 70}")

  a = results["nn.scan(axes)"]
  c = results["lax.scan(xs)"]
  d = results["lax.scan(closure)"]
  b = results["nn.scan(bcast)"]

  print(f"\n  nn.scan(axes) vs lax.scan(xs):")
  if a["while_loops"] == c["while_loops"] and a["grad_eqns"] == c["grad_eqns"]:
    print("    IDENTICAL structure -- scan mechanism is NOT the cause")
  else:
    print(
        f"    DIFFERENT: while={a['while_loops']} vs {c['while_loops']}, "
        f"grad_eqns={a['grad_eqns']} vs {c['grad_eqns']}"
    )

  print(f"\n  lax.scan(xs) vs lax.scan(closure):")
  if c["while_loops"] == d["while_loops"]:
    print(f"    Same while-loops ({c['while_loops']})")
  else:
    print(f"    DIFFERENT: xs={c['while_loops']} vs closure={d['while_loops']}")

  # Key finding: check num_consts
  print(f"\n  KEY: scan num_consts comparison (how params enter scan):")
  for name, res in results.items():
    scans = [e for e in res["grad_jaxpr"].jaxpr.eqns if e.primitive.name == "scan"]
    consts = [se.params.get("num_consts", "?") for se in scans]
    print(f"    {name}: num_consts per scan = {consts}")
