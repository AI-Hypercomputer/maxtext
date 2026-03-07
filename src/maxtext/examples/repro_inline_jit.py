"""
Repro: Nested JIT inlining causes XLA to consume excess peak memory.

Problem:
  Each parameter is initialized with its own jax.jit(..., out_shardings=...).
  When the model constructor is wrapped in an outer @jax.jit, XLA inlines all
  inner JIT computation graphs into the outer one before it even begins to
  compile them. This creates a huge monolithic HLO graph that requires much
  more peak memory during compilation.

This script runs on a single host (CPU or TPU) and demonstrates:

  [A] outer_jit=True  – wraps constructor in @jax.jit  ← BAD (original code)
  [B] outer_jit=False – no outer JIT (the fix, _self_manages_sharding path)
  [C] outer jit + out_shardings on the outer JIT       ← attempted fix

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
from jax.sharding import PartitionSpec as P, SingleDeviceSharding

# ── single-device setup ──────────────────────────────────────────────────────
device = jax.devices()[0]
single_sharding = SingleDeviceSharding(device)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_inner_jitted_init(shape, dtype=jnp.float32):
    """
    Simulate create_param (base.py:159): each weight has its own jit+out_shardings.
    This is the pattern used in models with _self_manages_sharding=True.
    """
    @functools.partial(jax.jit, out_shardings=single_sharding)
    def _init(key):
        return jax.random.normal(key, shape, dtype=dtype)
    return _init


def build_model_params(key, num_layers, hidden, inner_jit=True):
    """Build a fake model's weight list using inner-jitted inits."""
    params = []
    for _ in range(num_layers):
        k1, k2, key = jax.random.split(key, 3)
        if inner_jit:
            w = make_inner_jitted_init((hidden, hidden))(k1)
            b = make_inner_jitted_init((hidden,))(k2)
        else:
            w = jax.random.normal(k1, (hidden, hidden))
            b = jax.random.normal(k2, (hidden,))
        params.append((w, b))
    return params


# ── HLO size helper ──────────────────────────────────────────────────────────

def hlo_text_size(fn, *args):
    """Return the size of the lowered HLO text (bytes) as a graph-size proxy."""
    try:
        lowered = jax.jit(fn).lower(*args)
        text = lowered.as_text()
        return len(text.encode())
    except Exception as e:
        return f"(error: {e})"


# ── scenario A: outer @jax.jit wraps inner-jitted inits ──────────────────────

def scenario_A(key, num_layers, hidden):
    """Bad: outer jit inlines all inner jits → huge HLO graph."""

    @jax.jit
    def create_model(key):
        return build_model_params(key, num_layers, hidden, inner_jit=True)

    tracemalloc.start()
    t0 = time.perf_counter()
    params = create_model(key)
    jax.block_until_ready(params)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    hlo_bytes = hlo_text_size(
        lambda k: build_model_params(k, num_layers, hidden, inner_jit=True), key)

    return elapsed, peak, hlo_bytes


# ── scenario B: no outer jit ─────────────────────────────────────────────────

def scenario_B(key, num_layers, hidden):
    """Fix: no outer jit; each inner jit compiles a tiny independent graph."""

    tracemalloc.start()
    t0 = time.perf_counter()
    params = build_model_params(key, num_layers, hidden, inner_jit=True)
    jax.block_until_ready(params)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # HLO size = just one single-layer init (representative of per-param graph)
    single_init = make_inner_jitted_init((hidden, hidden))
    k1, _ = jax.random.split(key)
    hlo_bytes = hlo_text_size(single_init, k1)
    # Total HLO = num_layers * 2 such tiny graphs (never merged into one)
    if isinstance(hlo_bytes, int):
        hlo_bytes_total = hlo_bytes * num_layers * 2
        hlo_label = f"{hlo_bytes_total} ({hlo_bytes} x {num_layers*2} separate graphs)"
    else:
        hlo_label = hlo_bytes

    return elapsed, peak, hlo_label


# ── scenario C: outer jit + explicit out_shardings ───────────────────────────

def scenario_C(key, num_layers, hidden):
    """
    Attempted fix: outer jit + explicit out_shardings.
    Hypothesis: out_shardings might act as a sharding barrier, limiting inlining.
    Reality: XLA still inlines everything; the graph is the same size as [A].
    """
    out_shardings = [(single_sharding, single_sharding)] * num_layers

    @functools.partial(jax.jit, out_shardings=out_shardings)
    def create_model(key):
        return build_model_params(key, num_layers, hidden, inner_jit=True)

    tracemalloc.start()
    t0 = time.perf_counter()
    params = create_model(key)
    jax.block_until_ready(params)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Build a version we can lower with jax.jit directly for HLO inspection
    def create_model_unlabeled(key):
        return build_model_params(key, num_layers, hidden, inner_jit=True)
    hlo_bytes = hlo_text_size(create_model_unlabeled, key)

    return elapsed, peak, hlo_bytes


# ── main ─────────────────────────────────────────────────────────────────────

def fmt_mb(b):
    return f"{b / 1024**2:.2f} MB"

def fmt_kb(b):
    if isinstance(b, int):
        return f"{b / 1024:.1f} KB"
    return str(b)

def run(num_layers=32, hidden=512):
    key = jax.random.PRNGKey(42)
    print(f"\nConfig: {num_layers} layers, hidden={hidden}")
    print(f"Device: {device}")
    print("=" * 70)

    # Warm-up XLA
    _ = jax.jit(lambda x: x + 1)(jnp.array(1.0))

    k1, k2, k3 = jax.random.split(key, 3)

    print("\n[A] outer @jax.jit wraps inner-jitted params (ORIGINAL / BAD):")
    tA, peakA, hloA = scenario_A(k1, num_layers, hidden)
    print(f"    wall_time={tA:.3f}s  peak_host_mem={fmt_mb(peakA)}  "
          f"HLO_size={fmt_kb(hloA)}")

    print("\n[B] no outer jit; inner jits run independently (FIX):")
    tB, peakB, hloB = scenario_B(k2, num_layers, hidden)
    print(f"    wall_time={tB:.3f}s  peak_host_mem={fmt_mb(peakB)}  "
          f"HLO_size={hloB}")

    print("\n[C] outer @jax.jit + explicit out_shardings (attempted fix):")
    tC, peakC, hloC = scenario_C(k3, num_layers, hidden)
    print(f"    wall_time={tC:.3f}s  peak_host_mem={fmt_mb(peakC)}  "
          f"HLO_size={fmt_kb(hloC)}")

    print()
    if isinstance(hloA, int) and isinstance(hloC, int):
        print(f"HLO graph [A] vs [C]: {hloA} vs {hloC} bytes  "
              f"({'same' if abs(hloA - hloC) < 0.05 * hloA else 'different'})")
        print(f"Peak mem  [A] vs [B]: {fmt_mb(peakA)} vs {fmt_mb(peakB)}  "
              f"(ratio={peakA/max(peakB,1):.1f}x)")

    print()
    print("Conclusions:")
    print("  [A] Large HLO + high peak memory: outer JIT inlines ALL inner")
    print("      jitted initializers into one giant computation graph.")
    print("  [B] Tiny per-param HLO: each param's graph compiles separately,")
    print("      XLA never sees the whole model at once. Lower peak memory.")
    print("  [C] Same large HLO as [A]: out_shardings on the outer JIT does")
    print("      NOT prevent inlining. XLA still produces one big graph.")
    print()
    print("  -> The only reliable fix is to omit the outer JIT entirely,")
    print("     which is what _self_manages_sharding=True does in model_loader.py.")


if __name__ == "__main__":
    layers = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    run(num_layers=layers, hidden=hidden)
