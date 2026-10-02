# XLA:TPU — host-offload remat policies fail inside nested `lax.scan`

A `jax.checkpoint` policy that offloads residuals to pinned host works when the
checkpointed body sits inside a single `jax.lax.scan`. Put that scan inside a
second `lax.scan` and the compile fails with an internal post-optimization
error.

`nested_scan_host_offload.py` is self-contained (~100 lines, JAX only, no model
code) and runs on any TPU.

## Reproduce

```
python nested_scan_host_offload.py
```

Observed on jax/jaxlib `0.10.2`, libtpu `0.0.42.1`, v5p.

```
FLAT: compiled
NESTED: FAILED -- INTERNAL: during context [post-optimization]: The async-start
  expects the shape of operand 0 to match the async shape at index {0}
  (bf16[4,3,8,512,1024]{4,3,2,1,0:T(8,128)(2,1)}
   vs bf16[4,3,8,512,1024]{4,3,2,1,0:T(8,128)(2,1)S(5)}).
TRIP_COUNT_ONE: FAILED -- INTERNAL: during context [post-optimization]:
  %bitcast.31 = bf16[1,1,8,512,1024]{4,3,2,1,0:T(8,128)(2,1)S(5)} bitcast(%closed_call.6),
  metadata={op_name="jit(trip_count_one)/jvp()/while/body/jit(dynamic_update_index_in_dim)/broadcast_in_dim"}:
  Bitcast cannot have different memory spaces of output (5) and operand (0)
  (bf16[1,1,8,512,1024]{4,3,2,1,0:T(8,128)(2,1)S(5)})
  (bf16[1,8,512,1024]{3,2,1,0:T(8,128)(2,1)}).
```

## The three cases

All three run the same layer body, `tanh(x @ w)`, checkpointed with

```python
jax.checkpoint_policies.save_and_offload_only_these_names(
    names_which_can_be_saved=(),
    names_which_can_be_offloaded=("layer_input",),
    offload_src="device",
    offload_dst="pinned_host",
)
```

| case | loop structure | result |
| --- | --- | --- |
| `FLAT` | one `scan` over 12 layers | compiles |
| `NESTED` | `scan` over 4 blocks, each an inner `scan` over 3 layers | fails |
| `TRIP_COUNT_ONE` | `scan` over 12 layers, each an inner `scan` of `length=1` | fails |

`TRIP_COUNT_ONE` computes exactly the same thing as `FLAT`. The only difference
is a loop that executes once and produces nothing extra. That it fails is why we
read this as a limitation in how the offload pass walks loop structure, rather
than anything to do with the volume of data being moved.

It needs the `skip-simplify-while-loops_trip-count-one` frontend attribute to
reproduce, otherwise the simplifier deletes the loop and the nesting with it.
That attribute is not contrived: models set it deliberately to keep a
trip-count-one scan as a scheduling barrier around a single layer.

## What we think is happening

A residual that leaves a `lax.scan` body is not a value — it is a slice write
into a preallocated stacked accumulator, `dynamic-update-slice(acc, value, i, …)`,
where `acc` is a while-loop carry. For one scan level the host-offloader handles
this: the accumulator is placed in `S(5)` and each iteration DMAs into a slice.

With two levels the residual is stacked twice — into the inner accumulator, then
into the outer one — and the pass has to carry an `S(5)` buffer out of the inner
`while` as a tuple element, `dynamic-update-slice` it into an outer `S(5)`
accumulator of a different shape, and pair one `copy-start` in the inner body
with a `copy-done` in the outer backward. It appears to annotate only one side of
that boundary:

- `TRIP_COUNT_ONE` shows it directly. The bitcast's result was rewritten to
  `S(5)` and its operand was not. A bitcast is a pure shape reinterpretation, so
  a memory-space mismatch across one is unrepresentable — the pass constructed
  invalid HLO itself.
- `NESTED` shows the same thing at the async-copy pairing step.

## Why this matters

This is the layout any model with a heterogeneous layer cycle produces. Scanning
over *blocks*, where each block loops over its own layers, is how you scan a
repeating pattern of mixed layer types (linear-attention layers plus a periodic
full-attention layer, sliding-window plus global, and so on) without unrolling
the whole stack. Those are exactly the models where offloading activations is
most attractive, and today the combination cannot be compiled.

The only workaround available from the framework side is to flatten the inner
level into a Python loop so that just one scan level remains. That works, but it
unrolls the block body, which duplicates the rematerialized forward and costs
real HBM — on one 80B configuration, 7.2 GB, against 2.0 GB actually moved to
host. So the workaround is what makes offload unusable on these models, not the
offload itself.

## Ask

Support offloaded residuals stacked across nested loops. Failing that, a clean
error at lowering that names the nesting would at least be actionable — all
three symptoms we have seen surface as internal post-optimization crashes.

A third symptom appears in larger programs with more offloaded names, which we
have not reduced to a small repro:

```
INVALID_ARGUMENT: E1200: CompileTimeHostOffloadOutputLocationMismatch:
Tensor which is moved to host (starting from ...) is returned from the entry
computation but the layout for this output is not set to host memory.
```

That one looks like the same propagation overshooting instead of undershooting:
`S(5)` reaches an entry-computation result because the matching `MoveToDevice`
was never found across the nest.
