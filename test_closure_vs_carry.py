"""CHALLENGER-1: Test whether closure-captured vs carry-passed params actually
affect XLA loop decomposition / HLO while-loop count.

Hypothesis under attack:
  "Closure-captured params cause XLA to merge loops, while carry-passed
   params let XLA split them."

Three counter-arguments tested:
  1. When scan is UNROLLED (length=N, unroll=N), there is no while-loop at
     all, so loop decomposition is irrelevant.
  2. In the INNER scan (L2 microbatch scan), weights are passed as an
     argument to jax.vjp on BOTH sides, not closure-captured.
  3. The difference might be nn.scan vs jax.lax.scan lowering, not closure
     vs carry.

Variants tested:
  A) params_in_carry:   params passed through jax.lax.scan carry
  B) params_in_closure: params captured from outer scope (closure)
  C) params_via_nn_scan: params broadcast via nn.scan's variable_broadcast

For each variant we test TWO sub-cases:
  - scanned (unroll=1):  should produce a while-loop in HLO
  - unrolled (unroll=N): should NOT produce a while-loop

We compare:
  - Number of "while" ops in the HLO text
  - Total compiled memory estimate (if available)

Usage:
  python test_closure_vs_carry.py          # CPU fallback
  python test_closure_vs_carry.py --tpu    # TPU (real target)
"""

import argparse
import re
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

# Optional: Flax for variant C
try:
  from flax import linen as nn

  HAS_FLAX = True
except ImportError:
  HAS_FLAX = False
  print("[WARN] Flax not found. Variant C (nn.scan) will be skipped.")


# ---------------------------------------------------------------------------
# Configuration matching the real pipeline
# ---------------------------------------------------------------------------
NUM_WEIGHT_LEAVES = 9
WEIGHT_SHAPE = (2, 4096, 11008)  # (num_stages, hidden, intermediate)
DTYPE = jnp.float32

# Scan dimensions matching the real pipeline
OUTER_SCAN_LENGTH = 16  # num_pipeline_repeats * microbatches_per_stage
INNER_SCAN_LENGTH = 4  # num_pipeline_microbatches per stage

# Loop state: 5 leaves matching the real carry
STATE_SHAPES = {
    "state_io": (2, 2, 2, 512, 2048),  # stages, mbs_per_stage, mb_size, seq, embed
    "shift": (2, 2, 512, 2048),  # stages, mb_size, seq, embed
    "circ_storage": (2, 4, 2, 512, 2048),  # stages, num_mbs, mb_size, seq, embed
    "loop_iteration": (),  # scalar
    "output_accum": (2, 512, 2048),  # batch, seq, embed
}


def make_weights(rng):
  """Create 9 weight leaves matching real pipeline shape."""
  keys = jax.random.split(rng, NUM_WEIGHT_LEAVES)
  return tuple(jax.random.normal(k, WEIGHT_SHAPE, dtype=DTYPE) for k in keys)


def make_loop_state(rng):
  """Create loop state dict with 5 leaves."""
  keys = jax.random.split(rng, len(STATE_SHAPES))
  state = {}
  for (name, shape), key in zip(STATE_SHAPES.items(), keys):
    if shape == ():
      state[name] = jnp.int32(0)
    else:
      state[name] = jax.random.normal(key, shape, dtype=DTYPE)
  return state


def simple_stage_fn(state, weights_tuple):
  """Mimics a single pipeline stage: apply each weight leaf to state."""
  # Simple matmul-like operation to make XLA work non-trivially
  x = state["shift"]
  for w in weights_tuple:
    # Reduce over one dim: (2, 4096, 11008) applied to (..., 2048)
    # We'll just do a contraction to keep shapes reasonable
    x = x + jnp.sum(w[0, :2048, :2048], axis=-1)  # simplified
  new_state = dict(state)
  new_state["shift"] = x
  new_state["loop_iteration"] = state["loop_iteration"] + 1
  return new_state


# ---------------------------------------------------------------------------
# Variant A: params in carry
# ---------------------------------------------------------------------------
def variant_a_carry(loop_state, weights, length, unroll):
  """Params passed through scan carry alongside loop_state."""

  def body(carry, _):
    state, w = carry
    new_state = simple_stage_fn(state, w)
    return (new_state, w), None

  (final_state, _), _ = lax.scan(body, (loop_state, weights), None, length=length, unroll=unroll)
  return final_state


# ---------------------------------------------------------------------------
# Variant B: params in closure
# ---------------------------------------------------------------------------
def variant_b_closure(loop_state, weights, length, unroll):
  """Params captured from closure (outer scope), NOT in carry."""

  def body(state, _):
    new_state = simple_stage_fn(state, weights)  # weights from closure
    return new_state, None

  final_state, _ = lax.scan(body, loop_state, None, length=length, unroll=unroll)
  return final_state


# ---------------------------------------------------------------------------
# Variant C: params via nn.scan (Flax lifting)
# ---------------------------------------------------------------------------
if HAS_FLAX:

  class ScanModule(nn.Module):
    """Uses nn.scan with variable_broadcast to pass weights."""

    num_weight_leaves: int = NUM_WEIGHT_LEAVES

    @nn.compact
    def __call__(self, loop_state):
      # Create weight params (broadcast, not carried)
      weights = tuple(
          self.param(f"w_{i}", nn.initializers.lecun_normal(), WEIGHT_SHAPE) for i in range(self.num_weight_leaves)
      )

      def body_fn(module, carry, _):
        new_state = simple_stage_fn(carry, weights)
        return new_state, None

      # nn.scan with variable_broadcast
      ScanBody = nn.scan(
          nn.Module,
          variable_broadcast="params",
          split_rngs={"params": False},
          length=None,  # set at call time
      )
      # We can't easily use nn.scan this way, so we'll use a simpler approach

      return simple_stage_fn(loop_state, weights)

  def variant_c_nn_scan(loop_state, weights, length, unroll):
    """Uses nn.scan's variable_broadcast mechanism."""

    # nn.scan wraps a Module class, broadcasting params
    class StageModule(nn.Module):

      @nn.compact
      def __call__(self, state):
        ws = tuple(self.param(f"w_{i}", lambda k, s: weights[i], WEIGHT_SHAPE) for i in range(NUM_WEIGHT_LEAVES))
        return simple_stage_fn(state, ws)

    ScanStage = nn.scan(
        StageModule,
        variable_broadcast="params",
        split_rngs={"params": False},
        length=length,
        unroll=unroll,
    )

    class Wrapper(nn.Module):

      @nn.compact
      def __call__(self, state):
        stage = ScanStage(name="scan_stage")
        final, _ = stage(state, None)  # (carry, xs) -> scan
        return final

    # Initialize and apply
    rng = jax.random.PRNGKey(42)
    wrapper = Wrapper()

    # For nn.scan, the __call__ signature is (carry, x) -> (carry, y)
    # We need to restructure StageModule

    class StageModuleV2(nn.Module):

      @nn.compact
      def __call__(self, carry, x):
        ws = tuple(self.param(f"w_{i}", lambda k, s: weights[i], WEIGHT_SHAPE) for i in range(NUM_WEIGHT_LEAVES))
        return simple_stage_fn(carry, ws), None

    ScanStageV2 = nn.scan(
        StageModuleV2,
        variable_broadcast="params",
        split_rngs={"params": False},
        length=length,
        unroll=unroll,
    )

    class WrapperV2(nn.Module):

      @nn.compact
      def __call__(self, state):
        final, _ = ScanStageV2(name="scanned")(state, None)
        return final

    wrapper_v2 = WrapperV2()
    variables = wrapper_v2.init(rng, loop_state)
    return wrapper_v2.apply(variables, loop_state)


# ---------------------------------------------------------------------------
# HLO analysis utilities
# ---------------------------------------------------------------------------
def count_while_loops(hlo_text: str) -> int:
  """Count the number of while-loop ops in HLO text."""
  return len(re.findall(r"\bwhile\b", hlo_text, re.IGNORECASE))


def count_hlo_ops(hlo_text: str, op_name: str) -> int:
  """Count occurrences of a specific HLO op."""
  return len(re.findall(rf"\b{op_name}\b", hlo_text))


def get_hlo_and_stats(fn, *args, name=""):
  """Compile fn, extract HLO text, count while-loops, estimate memory."""
  print(f"\n{'='*70}")
  print(f"  Compiling: {name}")
  print(f"{'='*70}")

  t0 = time.time()
  lowered = jax.jit(fn).lower(*args)
  compiled = lowered.compile()
  compile_time = time.time() - t0

  # Get HLO text
  hlo_text = compiled.as_text()

  # Count while-loops
  num_while = count_while_loops(hlo_text)

  # Count other interesting ops
  num_all_gather = count_hlo_ops(hlo_text, "all-gather")
  num_reduce_scatter = count_hlo_ops(hlo_text, "reduce-scatter")
  num_dynamic_slice = count_hlo_ops(hlo_text, "dynamic-slice")
  num_dynamic_update = count_hlo_ops(hlo_text, "dynamic-update-slice")

  # Try to get memory stats
  try:
    mem_analysis = compiled.memory_analysis()
    peak_mem_gb = mem_analysis.peak_bytes / 1e9 if hasattr(mem_analysis, "peak_bytes") else None
  except Exception:
    peak_mem_gb = None

  # Also check cost_analysis
  try:
    cost = compiled.cost_analysis()
    if isinstance(cost, list) and len(cost) > 0:
      cost = cost[0]
    flops = cost.get("flops", None) if isinstance(cost, dict) else None
  except Exception:
    flops = None

  # HLO size
  hlo_lines = hlo_text.count("\n")

  print(f"  Compile time:        {compile_time:.2f}s")
  print(f"  HLO lines:           {hlo_lines}")
  print(f"  while-loop count:    {num_while}")
  print(f"  all-gather count:    {num_all_gather}")
  print(f"  reduce-scatter:      {num_reduce_scatter}")
  print(f"  dynamic-slice:       {num_dynamic_slice}")
  print(f"  dynamic-update-slice:{num_dynamic_update}")
  if peak_mem_gb is not None:
    print(f"  Peak memory (GB):    {peak_mem_gb:.3f}")
  if flops is not None:
    print(f"  FLOPs:               {flops:.2e}")

  return {
      "name": name,
      "num_while": num_while,
      "hlo_lines": hlo_lines,
      "peak_mem_gb": peak_mem_gb,
      "flops": flops,
      "compile_time": compile_time,
      "num_all_gather": num_all_gather,
      "num_reduce_scatter": num_reduce_scatter,
      "num_dynamic_slice": num_dynamic_slice,
      "num_dynamic_update": num_dynamic_update,
      "hlo_text": hlo_text,
  }


# ---------------------------------------------------------------------------
# INNER scan test: params always as arguments (like L2 microbatch scan)
# ---------------------------------------------------------------------------
def inner_scan_carry(state, weights, length, unroll):
  """Inner scan with weights in carry (like if BSW were carried)."""

  def body(carry, _):
    s, w = carry
    ns = simple_stage_fn(s, w)
    return (ns, w), None

  (final, _), _ = lax.scan(body, (state, weights), None, length=length, unroll=unroll)
  return final


def inner_scan_closure(state, weights, length, unroll):
  """Inner scan with weights from closure (like NNX L2)."""

  def body(s, _):
    return simple_stage_fn(s, weights), None

  final, _ = lax.scan(body, state, None, length=length, unroll=unroll)
  return final


def inner_scan_arg(state, weights, length, unroll):
  """Inner scan with weights passed as xs (broadcast)."""

  # Broadcast weights across scan steps
  def body(s, w):
    return simple_stage_fn(s, w), None

  # Stack weights to create scan input
  ws_stacked = jax.tree.map(lambda w: jnp.broadcast_to(w, (length,) + w.shape), weights)
  final, _ = lax.scan(body, state, ws_stacked, length=length, unroll=unroll)
  return final


# ---------------------------------------------------------------------------
# Nested scan test: outer UNROLLED + inner SCANNED (matches real pipeline)
# ---------------------------------------------------------------------------
def nested_carry(loop_state, weights, outer_len, inner_len):
  """Outer unrolled, inner scanned. Weights in OUTER carry."""

  def outer_body(carry, _):
    state, w = carry

    def inner_body(s, _):
      return simple_stage_fn(s, w), None

    final_s, _ = lax.scan(inner_body, state, None, length=inner_len)
    return (final_s, w), None

  # Unrolled outer (unroll=outer_len)
  (final, _), _ = lax.scan(outer_body, (loop_state, weights), None, length=outer_len, unroll=outer_len)
  return final


def nested_closure(loop_state, weights, outer_len, inner_len):
  """Outer unrolled, inner scanned. Weights in CLOSURE."""

  def outer_body(state, _):
    def inner_body(s, _):
      return simple_stage_fn(s, weights), None  # closure capture

    final_s, _ = lax.scan(inner_body, state, None, length=inner_len)
    return final_s, None

  # Unrolled outer (unroll=outer_len)
  final, _ = lax.scan(outer_body, loop_state, None, length=outer_len, unroll=outer_len)
  return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--tpu", action="store_true", help="Assert TPU backend")
  parser.add_argument("--small", action="store_true", help="Use smaller shapes for faster testing")
  parser.add_argument("--dump-hlo", action="store_true", help="Dump HLO text files for manual inspection")
  args = parser.parse_args()

  if args.tpu:
    backend = jax.devices()[0].platform
    assert backend == "tpu", f"Expected TPU, got {backend}"
    print(f"Running on TPU: {jax.devices()}")
  else:
    print(f"Running on: {jax.devices()[0].platform} " f"({len(jax.devices())} devices)")

  # Adjust shapes for fast CPU testing
  global WEIGHT_SHAPE, STATE_SHAPES
  if args.small:
    WEIGHT_SHAPE = (2, 64, 128)
    STATE_SHAPES = {
        "state_io": (2, 2, 2, 16, 64),
        "shift": (2, 2, 16, 64),
        "circ_storage": (2, 4, 2, 16, 64),
        "loop_iteration": (),
        "output_accum": (2, 16, 64),
    }

  rng = jax.random.PRNGKey(0)
  rng1, rng2 = jax.random.split(rng)
  weights = make_weights(rng1)
  loop_state = make_loop_state(rng2)

  weight_bytes = sum(w.nbytes for w in weights)
  state_bytes = sum(v.nbytes for v in jax.tree.leaves(loop_state))
  print(f"\nWeight leaves: {len(weights)}, total: {weight_bytes/1e9:.3f} GB")
  print(f"State leaves:  {len(jax.tree.leaves(loop_state))}, " f"total: {state_bytes/1e9:.3f} GB")

  results = {}

  # ====================================================================
  # Test 1: Single-level scan (outer scan only), SCANNED (unroll=1)
  # ====================================================================
  print("\n" + "#" * 70)
  print("# TEST 1: Single-level scan, SCANNED (unroll=1)")
  print("# This IS where while-loops should appear")
  print("#" * 70)

  r = get_hlo_and_stats(
      lambda s: variant_a_carry(s, weights, OUTER_SCAN_LENGTH, 1),
      loop_state,
      name="A: params_in_carry (scanned, unroll=1)",
  )
  results["A_scanned"] = r

  r = get_hlo_and_stats(
      lambda s: variant_b_closure(s, weights, OUTER_SCAN_LENGTH, 1),
      loop_state,
      name="B: params_in_closure (scanned, unroll=1)",
  )
  results["B_scanned"] = r

  # ====================================================================
  # Test 2: Single-level scan, UNROLLED (unroll=length)
  # ====================================================================
  print("\n" + "#" * 70)
  print("# TEST 2: Single-level scan, UNROLLED (unroll=length)")
  print("# NO while-loops expected — makes loop decomposition irrelevant")
  print("#" * 70)

  r = get_hlo_and_stats(
      lambda s: variant_a_carry(s, weights, OUTER_SCAN_LENGTH, OUTER_SCAN_LENGTH),
      loop_state,
      name="A: params_in_carry (unrolled)",
  )
  results["A_unrolled"] = r

  r = get_hlo_and_stats(
      lambda s: variant_b_closure(s, weights, OUTER_SCAN_LENGTH, OUTER_SCAN_LENGTH),
      loop_state,
      name="B: params_in_closure (unrolled)",
  )
  results["B_unrolled"] = r

  # ====================================================================
  # Test 3: Inner scan only (L2 microbatch scan, length=4)
  # ====================================================================
  print("\n" + "#" * 70)
  print("# TEST 3: Inner scan (L2 microbatch scan), scanned")
  print("# Both sides should be IDENTICAL — weights are arguments, not closure")
  print("#" * 70)

  r = get_hlo_and_stats(
      lambda s: inner_scan_carry(s, weights, INNER_SCAN_LENGTH, 1), loop_state, name="Inner: weights_in_carry (scanned)"
  )
  results["inner_carry"] = r

  r = get_hlo_and_stats(
      lambda s: inner_scan_closure(s, weights, INNER_SCAN_LENGTH, 1),
      loop_state,
      name="Inner: weights_in_closure (scanned)",
  )
  results["inner_closure"] = r

  # ====================================================================
  # Test 4: Nested scan (outer UNROLLED + inner SCANNED)
  #         This matches the real pipeline: scan_pipeline_repeats=False
  # ====================================================================
  print("\n" + "#" * 70)
  print("# TEST 4: Nested scan (outer UNROLLED, inner SCANNED)")
  print("# Matches real pipeline: outer unrolled, inner L2 scan")
  print("#" * 70)

  r = get_hlo_and_stats(
      lambda s: nested_carry(s, weights, OUTER_SCAN_LENGTH, INNER_SCAN_LENGTH),
      loop_state,
      name="Nested: weights_in_outer_carry",
  )
  results["nested_carry"] = r

  r = get_hlo_and_stats(
      lambda s: nested_closure(s, weights, OUTER_SCAN_LENGTH, INNER_SCAN_LENGTH),
      loop_state,
      name="Nested: weights_in_closure",
  )
  results["nested_closure"] = r

  # ====================================================================
  # Test 5: Variant C — nn.scan with variable_broadcast (if Flax available)
  # ====================================================================
  if HAS_FLAX:
    print("\n" + "#" * 70)
    print("# TEST 5: nn.scan with variable_broadcast")
    print("# Tests whether Flax lifting mechanism differs from lax.scan")
    print("#" * 70)

    try:
      r = get_hlo_and_stats(
          lambda s: variant_c_nn_scan(s, weights, OUTER_SCAN_LENGTH, 1),
          loop_state,
          name="C: nn.scan variable_broadcast (scanned)",
      )
      results["C_scanned"] = r

      r = get_hlo_and_stats(
          lambda s: variant_c_nn_scan(s, weights, OUTER_SCAN_LENGTH, OUTER_SCAN_LENGTH),
          loop_state,
          name="C: nn.scan variable_broadcast (unrolled)",
      )
      results["C_unrolled"] = r
    except Exception as e:
      print(f"  [SKIP] nn.scan variant failed: {e}")

  # ====================================================================
  # Summary
  # ====================================================================
  print("\n" + "=" * 70)
  print("SUMMARY: while-loop counts and memory")
  print("=" * 70)
  print(f"{'Variant':<45} {'while':>6} {'HLO':>7} {'Mem(GB)':>9}")
  print("-" * 70)
  for key, r in results.items():
    mem_str = f"{r['peak_mem_gb']:.3f}" if r["peak_mem_gb"] else "N/A"
    print(f"{r['name']:<45} {r['num_while']:>6} " f"{r['hlo_lines']:>7} {mem_str:>9}")

  # ====================================================================
  # Diagnosis
  # ====================================================================
  print("\n" + "=" * 70)
  print("DIAGNOSIS")
  print("=" * 70)

  # Check counter-argument 1: unrolled scans have no while-loops
  a_un = results.get("A_unrolled", {}).get("num_while", -1)
  b_un = results.get("B_unrolled", {}).get("num_while", -1)
  if a_un == 0 and b_un == 0:
    print("[CONFIRMED] Counter-arg 1: Unrolled scans have 0 while-loops.")
    print("  => When scan_pipeline_repeats=False, outer loop is unrolled,")
    print("     so XLA loop decomposition of outer scan is IRRELEVANT.")
  elif a_un == b_un:
    print(f"[PARTIAL] Counter-arg 1: Both unrolled variants have " f"{a_un} while-loops (expected 0).")
  else:
    print(f"[UNEXPECTED] Counter-arg 1: carry={a_un}, closure={b_un}")

  # Check counter-argument 2: inner scan carry vs closure identical
  ic = results.get("inner_carry", {}).get("num_while", -1)
  icl = results.get("inner_closure", {}).get("num_while", -1)
  if ic == icl:
    print(f"\n[CONFIRMED] Counter-arg 2: Inner scan carry ({ic}) == " f"closure ({icl}) while-loops.")
    print("  => L2 microbatch scan structure is IDENTICAL regardless")
    print("     of where weights come from.")
  else:
    print(f"\n[UNEXPECTED] Counter-arg 2: Inner carry={ic} vs " f"closure={icl}")

  # Check key question: does closure vs carry matter for scanned loops?
  as_ = results.get("A_scanned", {}).get("num_while", -1)
  bs = results.get("B_scanned", {}).get("num_while", -1)
  if as_ == bs:
    print(f"\n[KEY FINDING] Scanned single-level: carry ({as_}) == " f"closure ({bs}) while-loops.")
    print("  => Closure vs carry does NOT affect XLA loop decomposition!")
    print("  => The hypothesis 'closure-captured params prevent loop")
    print("     decomposition' is FALSIFIED.")
  else:
    print(f"\n[KEY FINDING] Scanned single-level: carry ({as_}) != " f"closure ({bs}).")
    print("  => There IS a difference in while-loop count!")
    print("  => Needs further investigation of HLO structure.")

  # Check nested scan
  nc = results.get("nested_carry", {}).get("num_while", -1)
  ncl = results.get("nested_closure", {}).get("num_while", -1)
  print(f"\n[NESTED] carry={nc} vs closure={ncl} while-loops.")
  if nc == ncl:
    print("  => Nested structure also identical — closure is NOT the issue.")
  else:
    print("  => Nested structure differs — investigate further.")

  # Memory comparison
  print("\n[MEMORY COMPARISON]")
  for key in ["A_scanned", "B_scanned", "nested_carry", "nested_closure"]:
    r = results.get(key)
    if r and r["peak_mem_gb"] is not None:
      print(f"  {r['name']:<45}: {r['peak_mem_gb']:.3f} GB")
    elif r:
      print(f"  {r['name']:<45}: N/A (use --tpu for memory stats)")

  # Dump HLO if requested
  if args.dump_hlo:
    for key, r in results.items():
      fname = f"hlo_{key}.txt"
      with open(fname, "w") as f:
        f.write(r["hlo_text"])
      print(f"  Dumped HLO to {fname}")

  print("\n" + "=" * 70)
  print("CONCLUSION")
  print("=" * 70)
  print("If all while-loop counts match between carry and closure variants,")
  print("then the hypothesis 'closure-captured state prevents XLA loop")
  print("decomposition' is FALSIFIED. The 6.2 GB memory gap between Linen")
  print("and NNX must come from a different source — likely:")
  print("  1. nn.scan vs jax.lax.scan lowering differences")
  print("  2. custom_vjp residual structure differences")
  print("  3. Checkpoint/remat policy application differences")
  print("  4. Weight prefetching double-buffer allocation patterns")


if __name__ == "__main__":
  main()
