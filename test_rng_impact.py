"""Test: Does number of RNG streams in a scan body affect XLA memory?

Measures peak memory with 0, 1, 9, and 18 RNG fold_in operations per
scan iteration to test the claim that 18 RNG mutables in NNX cause
measurably higher memory than Linen's single split_rngs stream.

Also compares fold_in (NNX approach) vs split (Linen nn.scan approach).
"""

import jax
import jax.numpy as jnp
import jax.random
import time


def measure_scan_rng(num_rng_streams: int, num_iterations: int = 32, use_fold_in: bool = True, hidden_dim: int = 2048):
  """Run a scan with N RNG fold_in/split ops per iteration, measure memory."""

  key = jax.random.PRNGKey(0)
  keys = [jax.random.PRNGKey(i) for i in range(num_rng_streams)]

  # Simulated state: a matrix that gets updated each iteration
  init_state = jnp.zeros((hidden_dim, hidden_dim), dtype=jnp.bfloat16)
  # Simulated weight
  weight = jax.random.normal(key, (hidden_dim, hidden_dim), dtype=jnp.bfloat16)

  if num_rng_streams == 0:
    # No RNG at all
    def scan_body(carry, _):
      state = carry
      new_state = state + jnp.dot(state, weight) * 0.001
      return new_state, None

  elif use_fold_in:
    # NNX approach: fold_in on each key
    def scan_body(carry, xs):
      state, iteration = carry[0], carry[1]
      # Fold iteration into each RNG key (what _advance_rng_state does)
      derived_keys = [jax.random.fold_in(k, iteration) for k in keys]
      # Use first derived key for dropout-like mask
      mask = jax.random.bernoulli(derived_keys[0], 0.9, shape=state.shape).astype(jnp.bfloat16)
      new_state = state + jnp.dot(state * mask, weight) * 0.001
      return (new_state, iteration + 1), None

  else:
    # Linen approach: single split per iteration
    def scan_body(carry, xs):
      state, rng = carry[0], carry[1]
      rng, dropout_rng = jax.random.split(rng)
      mask = jax.random.bernoulli(dropout_rng, 0.9, shape=state.shape).astype(jnp.bfloat16)
      new_state = state + jnp.dot(state * mask, weight) * 0.001
      return (new_state, rng), None

  if num_rng_streams == 0:
    init_carry = init_state
  elif use_fold_in:
    init_carry = (init_state, jnp.int32(0))
  else:
    init_carry = (init_state, key)

  @jax.jit
  def run():
    return jax.lax.scan(scan_body, init_carry, None, length=num_iterations)

  # Warmup
  result = run()
  jax.block_until_ready(result)

  # Measure
  t0 = time.perf_counter()
  result = run()
  jax.block_until_ready(result)
  t1 = time.perf_counter()

  return t1 - t0


def measure_scan_rng_with_grad(
    num_rng_streams: int, num_iterations: int = 32, use_fold_in: bool = True, hidden_dim: int = 2048
):
  """Same as above but through jax.grad to measure backward pass memory impact."""

  key = jax.random.PRNGKey(0)
  keys = tuple(jax.random.PRNGKey(i) for i in range(max(num_rng_streams, 1)))

  init_state = jnp.zeros((hidden_dim, hidden_dim), dtype=jnp.bfloat16)
  weight = jax.random.normal(key, (hidden_dim, hidden_dim), dtype=jnp.bfloat16)

  if num_rng_streams == 0:

    def scan_fn(state, weight):
      def body(carry, _):
        s = carry
        s = s + jnp.dot(s, weight) * 0.001
        return s, None

      final, _ = jax.lax.scan(body, state, None, length=num_iterations)
      return jnp.sum(final)

  elif use_fold_in:

    def scan_fn(state, weight):
      def body(carry, _):
        s, it = carry
        derived = [jax.random.fold_in(k, it) for k in keys[:num_rng_streams]]
        mask = jax.random.bernoulli(derived[0], 0.9, shape=s.shape).astype(jnp.bfloat16)
        s = s + jnp.dot(s * mask, weight) * 0.001
        return (s, it + 1), None

      (final, _), _ = jax.lax.scan(body, (state, jnp.int32(0)), None, length=num_iterations)
      return jnp.sum(final)

  else:

    def scan_fn(state, weight):
      def body(carry, _):
        s, rng = carry
        rng, use_rng = jax.random.split(rng)
        mask = jax.random.bernoulli(use_rng, 0.9, shape=s.shape).astype(jnp.bfloat16)
        s = s + jnp.dot(s * mask, weight) * 0.001
        return (s, rng), None

      (final, _), _ = jax.lax.scan(body, (state, key), None, length=num_iterations)
      return jnp.sum(final)

  grad_fn = jax.jit(jax.grad(scan_fn, argnums=(0, 1)))

  # Warmup
  grads = grad_fn(init_state, weight)
  jax.block_until_ready(grads)

  # Measure
  t0 = time.perf_counter()
  grads = grad_fn(init_state, weight)
  jax.block_until_ready(grads)
  t1 = time.perf_counter()

  return t1 - t0


def count_hlo_ops(num_rng_streams: int, num_iterations: int = 32, use_fold_in: bool = True, hidden_dim: int = 512):
  """Count fold_in / threefry ops in HLO to verify jaxpr complexity claim."""
  key = jax.random.PRNGKey(0)
  keys = tuple(jax.random.PRNGKey(i) for i in range(max(num_rng_streams, 1)))

  init_state = jnp.zeros((hidden_dim, hidden_dim), dtype=jnp.bfloat16)
  weight = jax.random.normal(key, (hidden_dim, hidden_dim), dtype=jnp.bfloat16)

  if num_rng_streams == 0:

    def scan_fn(state, weight):
      def body(carry, _):
        return carry + jnp.dot(carry, weight) * 0.001, None

      final, _ = jax.lax.scan(body, state, None, length=num_iterations)
      return jnp.sum(final)

  elif use_fold_in:

    def scan_fn(state, weight):
      def body(carry, _):
        s, it = carry
        derived = [jax.random.fold_in(k, it) for k in keys[:num_rng_streams]]
        mask = jax.random.bernoulli(derived[0], 0.9, shape=s.shape).astype(jnp.bfloat16)
        s = s + jnp.dot(s * mask, weight) * 0.001
        return (s, it + 1), None

      (final, _), _ = jax.lax.scan(body, (state, jnp.int32(0)), None, length=num_iterations)
      return jnp.sum(final)

  else:

    def scan_fn(state, weight):
      def body(carry, _):
        s, rng = carry
        rng, use_rng = jax.random.split(rng)
        mask = jax.random.bernoulli(use_rng, 0.9, shape=s.shape).astype(jnp.bfloat16)
        s = s + jnp.dot(s * mask, weight) * 0.001
        return (s, rng), None

      (final, _), _ = jax.lax.scan(body, (state, key), None, length=num_iterations)
      return jnp.sum(final)

  grad_fn = jax.grad(scan_fn, argnums=(0, 1))
  lowered = jax.jit(grad_fn).lower(init_state, weight)
  hlo_text = lowered.as_text()

  # Count relevant ops
  fold_in_count = hlo_text.count("fold_in") + hlo_text.count("threefry")
  rng_bits_count = hlo_text.count("rng-bit-generator") + hlo_text.count("RngBitGenerator")
  dynamic_slice_count = hlo_text.count("dynamic-slice") + hlo_text.count("DynamicSlice")

  return {
      "fold_in_or_threefry": fold_in_count,
      "rng_bit_generator": rng_bits_count,
      "dynamic_slice": dynamic_slice_count,
      "hlo_lines": len(hlo_text.splitlines()),
  }


if __name__ == "__main__":
  print("=" * 70)
  print("TEST: RNG Stream Count Impact on Scan Body")
  print("=" * 70)

  # Part 1: Forward-only timing
  print("\n--- Forward-only timing (32 iterations, 2048x2048) ---")
  for n_rng in [0, 1, 9, 18]:
    if n_rng == 0:
      t = measure_scan_rng(n_rng)
      print(f"  {n_rng:2d} RNG streams (no RNG):    {t*1000:.2f} ms")
    else:
      t_fold = measure_scan_rng(n_rng, use_fold_in=True)
      print(f"  {n_rng:2d} RNG streams (fold_in):   {t_fold*1000:.2f} ms")
  t_split = measure_scan_rng(1, use_fold_in=False)
  print(f"   1 RNG stream  (split):     {t_split*1000:.2f} ms")

  # Part 2: Forward+backward timing
  print("\n--- Forward+Backward timing (grad, 32 iterations, 2048x2048) ---")
  for n_rng in [0, 1, 9, 18]:
    if n_rng == 0:
      t = measure_scan_rng_with_grad(n_rng)
      print(f"  {n_rng:2d} RNG streams (no RNG):    {t*1000:.2f} ms")
    else:
      t_fold = measure_scan_rng_with_grad(n_rng, use_fold_in=True)
      print(f"  {n_rng:2d} RNG streams (fold_in):   {t_fold*1000:.2f} ms")
  t_split = measure_scan_rng_with_grad(1, use_fold_in=False)
  print(f"   1 RNG stream  (split):     {t_split*1000:.2f} ms")

  # Part 3: HLO complexity
  print("\n--- HLO Op Counts (grad, 32 iterations, 512x512) ---")
  for n_rng in [0, 1, 9, 18]:
    if n_rng == 0:
      ops = count_hlo_ops(n_rng)
      print(f"  {n_rng:2d} RNG (none):   {ops}")
    else:
      ops = count_hlo_ops(n_rng, use_fold_in=True)
      print(f"  {n_rng:2d} RNG (fold_in): {ops}")
  ops_split = count_hlo_ops(1, use_fold_in=False)
  print(f"   1 RNG (split):  {ops_split}")

  print("\n" + "=" * 70)
  print("ANALYSIS:")
  print("If 18 RNG streams cause materially different memory/time vs 1,")
  print("the claim that RNG mutables are 'the only structural difference'")
  print("AND are significant would be supported.")
  print("If results are similar, RNG count is irrelevant to the memory gap.")
  print("=" * 70)
