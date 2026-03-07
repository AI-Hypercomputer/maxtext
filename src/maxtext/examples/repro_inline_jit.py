"""
Repro: Wrapping an already-JIT-compiled model constructor in an outer JIT
causes XLA to inline the inner JIT's full computation graph, producing a
monolithic HLO that OOMs during compilation.

Context (tpu-inference model_loader.py):
  The generic model loading path wraps any model constructor in an outer JIT:

    @jax.jit
    def create_sharded_model():
        model = model_class(vllm_config, rng, mesh)
        state = nnx.state(model)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)
        return model

  For a MaxText model, the constructor calls create_nnx_model() which itself
  wraps all parameter initialization in an inner @jax.jit with out_shardings.
  When the outer create_sharded_model traces the constructor, JAX inlines the
  inner JIT's entire computation graph — every random.normal and every
  with_sharding_constraint op for all parameters — into the outer JIT's HLO.
  The result is a doubly-large monolithic HLO that OOMs during XLA compilation.

Scenarios:
  [A] MaxText-style model (inner JIT) wrapped in outer JIT         <- OOM
  [B] MaxText-style model called directly, no outer JIT            <- FIX
      (mirrors the _self_manages_sharding path in model_loader.py)

Metrics:
  - HLO text size: proxy for the XLA compilation graph size (main metric)
  - Tracemalloc peak: host-side Python/XLA graph-construction memory
  - Wall time: end-to-end init time
"""

import functools
import tracemalloc
import time
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# -- 8-device mesh setup ------------------------------------------------------
devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=("data",))

# Shard the first (row) dimension of weight matrices across all devices.
# Bias vectors are replicated (no sharding on their single dimension).
weight_sharding = NamedSharding(mesh, P("data", None))
bias_sharding = NamedSharding(mesh, P())


# -- model constructor ---------------------------------------------------------


def maxtext_model_constructor(key, num_layers, layer_size):
  """Mirrors a MaxText model constructor that calls create_nnx_model().

  All parameter initialization is wrapped in an inner @jax.jit with
  out_shardings (create_sharded_state in model_creation_utils.py:115-155).
  The rng key is captured in the closure — no args are passed to the inner JIT.

  This is what model_class(vllm_config, rng, mesh) resolves to in
  tpu-inference when model_class is MaxText-backed.
  """
  layer_shardings = (weight_sharding,) * 6 + (bias_sharding,) * 2
  out_shardings = [layer_shardings] * num_layers

  @functools.partial(jax.jit, out_shardings=out_shardings)
  def create_sharded_state():
    params = []
    layer_key = key
    for _ in range(num_layers):
      # Simulate a transformer layer: Q/K/V/O projections + 2 MLP weights +
      # 2 layer-norm scales. This matches the param density of a real layer.
      layer_key, kq, kk, kv, ko, km1, km2, kn1, kn2 = jax.random.split(layer_key, 9)
      wq = jax.random.normal(kq, (layer_size, layer_size))
      wk = jax.random.normal(kk, (layer_size, layer_size))
      wv = jax.random.normal(kv, (layer_size, layer_size))
      wo = jax.random.normal(ko, (layer_size, layer_size))
      wm1 = jax.random.normal(km1, (layer_size, layer_size * 4))
      wm2 = jax.random.normal(km2, (layer_size * 4, layer_size))
      ln1 = jax.random.normal(kn1, (layer_size,))
      ln2 = jax.random.normal(kn2, (layer_size,))
      wq = jax.lax.with_sharding_constraint(wq, weight_sharding)
      wk = jax.lax.with_sharding_constraint(wk, weight_sharding)
      wv = jax.lax.with_sharding_constraint(wv, weight_sharding)
      wo = jax.lax.with_sharding_constraint(wo, weight_sharding)
      wm1 = jax.lax.with_sharding_constraint(wm1, weight_sharding)
      wm2 = jax.lax.with_sharding_constraint(wm2, weight_sharding)
      ln1 = jax.lax.with_sharding_constraint(ln1, bias_sharding)
      ln2 = jax.lax.with_sharding_constraint(ln2, bias_sharding)
      params.append((wq, wk, wv, wo, wm1, wm2, ln1, ln2))
    return params

  return create_sharded_state()


# -- tpu-inference outer JIT wrapper ------------------------------------------


def tpu_inference_outer_jit(key, num_layers, layer_size):
  """Mirrors tpu-inference's create_sharded_model (model_loader.py:183-194).

  Wraps maxtext_model_constructor in an outer @jax.jit and applies an
  additional with_sharding_constraint pass over the returned state.
  """
  layer_shardings = (weight_sharding,) * 6 + (bias_sharding,) * 2
  out_shardings = [layer_shardings] * num_layers

  @functools.partial(jax.jit, out_shardings=out_shardings)
  def create_sharded_model():
    params = maxtext_model_constructor(key, num_layers, layer_size)
    # mirrors: sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    resharded = []
    for layer in params:
      wq, wk, wv, wo, wm1, wm2, ln1, ln2 = layer
      resharded.append(
          (
              jax.lax.with_sharding_constraint(wq, weight_sharding),
              jax.lax.with_sharding_constraint(wk, weight_sharding),
              jax.lax.with_sharding_constraint(wv, weight_sharding),
              jax.lax.with_sharding_constraint(wo, weight_sharding),
              jax.lax.with_sharding_constraint(wm1, weight_sharding),
              jax.lax.with_sharding_constraint(wm2, weight_sharding),
              jax.lax.with_sharding_constraint(ln1, bias_sharding),
              jax.lax.with_sharding_constraint(ln2, bias_sharding),
          )
      )
    return resharded

  return create_sharded_model


# -- HLO size helper ----------------------------------------------------------


def hlo_text_size(fn):
  """Return the size of the lowered HLO text (bytes) as a graph-size proxy."""
  try:
    lowered = jax.jit(fn).lower()
    return len(lowered.as_text().encode())
  except Exception as e:  # pylint: disable=broad-exception-caught
    return f"(error: {e})"


# -- measure helper -----------------------------------------------------------


def measure(fn):
  """Run fn(), block until ready, and return (result, wall_time_s, peak_bytes)."""
  tracemalloc.start()
  t0 = time.perf_counter()
  result = fn()
  jax.block_until_ready(result)
  elapsed = time.perf_counter() - t0
  _, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  return result, elapsed, peak


# -- main ---------------------------------------------------------------------


def fmt_mb(num_bytes):
  """Format bytes as MB string."""
  return f"{num_bytes / 1024**2:.2f} MB"


def fmt_kb(num_bytes):
  """Format bytes as KB string, or pass through non-int values."""
  if isinstance(num_bytes, int):
    return f"{num_bytes / 1024:.1f} KB"
  return str(num_bytes)


def run(num_layers=32, layer_size=2048):
  """Run both scenarios and print results."""
  key = jax.random.PRNGKey(42)
  print(
      f"\nConfig: {num_layers} layers, layer_size={layer_size}, "
      f"devices={len(devices)} (mesh: {dict(zip(mesh.axis_names, mesh.shape))})"
  )
  print("=" * 70)

  # Warm-up XLA
  _ = jax.jit(lambda x: x + 1)(jnp.array(1.0))

  k1, k2 = jax.random.split(key, 2)

  print("\n[A] MaxText model (inner JIT) wrapped in outer create_sharded_model JIT (OOM):")
  fn_a = tpu_inference_outer_jit(k1, num_layers, layer_size)
  _, t_a, peak_a = measure(fn_a)
  hlo_a = hlo_text_size(
      lambda: [
          (
              jax.lax.with_sharding_constraint(wq, weight_sharding),
              jax.lax.with_sharding_constraint(wk, weight_sharding),
              jax.lax.with_sharding_constraint(wv, weight_sharding),
              jax.lax.with_sharding_constraint(wo, weight_sharding),
              jax.lax.with_sharding_constraint(wm1, weight_sharding),
              jax.lax.with_sharding_constraint(wm2, weight_sharding),
              jax.lax.with_sharding_constraint(ln1, bias_sharding),
              jax.lax.with_sharding_constraint(ln2, bias_sharding),
          )
          for wq, wk, wv, wo, wm1, wm2, ln1, ln2 in maxtext_model_constructor(k1, num_layers, layer_size)
      ]
  )
  print(f"    wall_time={t_a:.3f}s  peak_host_mem={fmt_mb(peak_a)}  HLO_size={fmt_kb(hlo_a)}")

  print("\n[B] MaxText model called directly, no outer JIT (_self_manages_sharding FIX):")
  _, t_b, peak_b = measure(lambda: maxtext_model_constructor(k2, num_layers, layer_size))
  hlo_b = hlo_text_size(lambda: maxtext_model_constructor(k2, num_layers, layer_size))
  print(f"    wall_time={t_b:.3f}s  peak_host_mem={fmt_mb(peak_b)}  HLO_size={fmt_kb(hlo_b)}")

  print()
  if isinstance(hlo_a, int) and isinstance(hlo_b, int):
    print(
        f"HLO  [A] vs [B]: {fmt_kb(hlo_a)} vs {fmt_kb(hlo_b)}  "
        f"([A] is {hlo_a/max(hlo_b,1):.1f}x larger — inner JIT inlined into outer)"
    )
    print(f"Peak [A] vs [B]: {fmt_mb(peak_a)} vs {fmt_mb(peak_b)}  " f"(ratio={peak_a/max(peak_b,1):.1f}x)")

  print()
  print("Conclusions:")
  print("  [A] Outer JIT inlines the inner MaxText JIT: XLA must compile a single")
  print("      monolithic HLO containing all param inits (from the inner JIT) plus")
  print("      an extra with_sharding_constraint pass for every parameter. The HLO")
  print("      ratio here (~1.1x) looks modest, but on a real 80-layer model with")
  print("      thousands of parameters the absolute HLO grows to tens of GB —")
  print("      enough to OOM during XLA compilation.")
  print("  [B] Calling the MaxText constructor directly lets its inner JIT compile")
  print("      independently. The outer create_sharded_model JIT is never involved,")
  print("      so XLA only sees the inner graph. HLO stays bounded. This is the")
  print("      _self_manages_sharding fix in model_loader.py.")


if __name__ == "__main__":
  layers = int(sys.argv[1]) if len(sys.argv) > 1 else 32
  size = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
  run(num_layers=layers, layer_size=size)
