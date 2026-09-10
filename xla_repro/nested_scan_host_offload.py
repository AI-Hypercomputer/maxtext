"""Minimal repro: `jax.checkpoint` host-offload policies crash XLA:TPU when the
offloaded residual is produced inside a *nested* `jax.lax.scan`.

A remat policy built with `save_and_offload_only_these_names(...,
offload_dst="pinned_host")` emits MoveToHost/MoveToDevice around the named
residual. When the checkpointed body sits inside a single `lax.scan`, XLA
handles this: the stacked residual accumulator lives in memory space S(5) and
each iteration DMAs into a slice of it. This is the ordinary "scan over
transformer layers + offload activations" pattern and it compiles fine.

Wrap that scan in a second `lax.scan` -- as any model with a heterogeneous layer
cycle does, scanning over *blocks* where each block loops over its own layers --
and the same policy fails to compile with an internal post-optimization error.
The residual is now stacked twice: `dynamic-update-slice` into the inner
accumulator, then again into the outer one. XLA's host-offloader appears to
propagate the S(5) annotation across only one of those levels.

Run:
    python nested_scan_host_offload.py

Expected output: FLAT passes, NESTED and TRIP_COUNT_ONE fail.

The TRIP_COUNT_ONE case is the same program as FLAT plus an inner `lax.scan` of
`length=1` -- a loop that runs exactly once and computes nothing extra. It is
enough to trigger the failure, which is why we believe this is a structural
limitation of the offload pass rather than anything to do with the amount of
data being moved.

Observed on jax/jaxlib 0.10.2, libtpu 0.0.42.1.
"""

import sys

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import xla_metadata

# Shapes are only large enough to make the offload worth doing; the failure does
# not depend on them.
BATCH, SEQ, DIM = 8, 512, 1024
NUM_BLOCKS, LAYERS_PER_BLOCK = 4, 3

OFFLOAD_POLICY = jax.checkpoint_policies.save_and_offload_only_these_names(
    names_which_can_be_saved=(),
    names_which_can_be_offloaded=("layer_input",),
    offload_src="device",
    offload_dst="pinned_host",
)


def layer(carry, weight):
  """One 'transformer layer'. Its input is the tensor we ask to be offloaded."""
  carry = checkpoint_name(carry, "layer_input")
  return jnp.tanh(carry @ weight), None


def flat(x, weights):
  """Single scan over all layers -- the homogeneous-model layout. Compiles."""
  body = jax.checkpoint(layer, policy=OFFLOAD_POLICY)
  out, _ = jax.lax.scan(body, x, weights)
  return jnp.sum(out)


def nested(x, weights):
  """Scan over blocks, each block scans over its own layers. Fails."""
  body = jax.checkpoint(layer, policy=OFFLOAD_POLICY)

  def block(carry, block_weights):
    out, _ = jax.lax.scan(body, carry, block_weights)
    return out, None

  # weights: [NUM_BLOCKS * LAYERS_PER_BLOCK, DIM, DIM] -> [NUM_BLOCKS, LAYERS_PER_BLOCK, ...]
  grouped = weights.reshape(NUM_BLOCKS, LAYERS_PER_BLOCK, *weights.shape[1:])
  out, _ = jax.lax.scan(block, x, grouped)
  return jnp.sum(out)


def trip_count_one(x, weights):
  """Identical to `flat`, plus an inner scan of `length=1` around each layer.

  The inner loop executes exactly once and computes nothing extra, so this is
  the same computation as `flat` -- only the loop structure differs. It still
  fails, which is why we read this as a structural limitation of the offload
  pass rather than anything to do with how much data is being moved.

  `skip-simplify-while-loops_trip-count-one` keeps XLA from deleting the loop.
  Real models use a trip-count-one scan deliberately, as a scheduling barrier
  around a single layer, and set this attribute for exactly that reason; without
  it the simplifier removes the loop and the nesting -- and the bug -- with it.
  """
  body = jax.checkpoint(layer, policy=OFFLOAD_POLICY)

  def outer_body(carry, weight):
    with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
      out, _ = jax.lax.scan(body, carry, weight[None], length=1)
    return out, None

  out, _ = jax.lax.scan(outer_body, x, weights)
  return jnp.sum(out)


CASES = {"FLAT": flat, "NESTED": nested, "TRIP_COUNT_ONE": trip_count_one}


def main():
  x = jnp.zeros((BATCH, SEQ, DIM), jnp.bfloat16)
  weights = jnp.zeros((NUM_BLOCKS * LAYERS_PER_BLOCK, DIM, DIM), jnp.bfloat16)

  failures = 0
  for name, fn in CASES.items():
    # Offloading only has an effect through the backward pass, which is what
    # consumes the rematerialization residuals.
    grad_fn = jax.jit(jax.grad(fn, argnums=1))
    try:
      grad_fn.lower(x, weights).compile()
      print(f"{name}: compiled")
    except Exception as e:  # pylint: disable=broad-except
      failures += 1
      first_line = str(e).strip().splitlines()[0]
      print(f"{name}: FAILED -- {first_line}")

  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
