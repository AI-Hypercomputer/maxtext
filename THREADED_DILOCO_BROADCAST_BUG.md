# Threaded DiLoCo `broadcast_params` layout mismatch

## Summary

The non-SPMD threaded DiLoCo syncer fails at its initial `broadcast_params` call. PJRT reports that the actual input tensor has **null tiling**, while the compiled executable expects TPU device layout `T(128)`.

This is a physical input-ABI mismatch at the learner-to-syncer transfer boundary, not a DiLoCo algorithm error and not a `PartitionSpec` mismatch. `T(128)` is TPU device-memory layout metadata in HLO. The broadcast JIT froze its input using `in_shardings` derived from the abstract model, while its concrete value arrived through a TPU-to-colocated-CPU transfer (or checkpoint restore) with a null CPU layout.

## Affected path

1. A learner owns TPU-sharded parameters.
2. `LearnerTransport.send_to_syncer` places them on the learner's colocated CPU submesh.
3. The syncer places the received tree on `global_cpu_mesh` using shardings derived from the abstract model.
4. `broadcast_params` has strict explicit `in_shardings` and broadcasts a leading `diloco` dimension.
5. PJRT rejects the call because actual and compiled device-local layouts differ.

There are two related hazards:

- A restored syncer state is used directly and is not explicitly re-placed onto the current `params_shardings` before the initial broadcast.
- `cpu_clone` (`jnp.copy`) is applied before the later `broadcast_frag` JIT. This creates another transfer into a JIT with a strict input contract and may lose the producer-selected device layout.

## Evidence and CPU coverage

- The failing JIT is built from shardings extracted from abstract parameters, while its concrete input comes from learner transfer or checkpoint restore.
- Reference TPU HLO files in this repository show `T(128)` in `entry_computation_layout`; it is a physical TPU tile, distinct from `NamedSharding.spec`.
- The old CPU integration test skips when JAX starts with one CPU device, so standard CPU CI can avoid this path.
- CPU PJRT does not implement TPU `T(128)` layouts. CPU can cover the sharding/lowering pipeline, but cannot faithfully reproduce the exact error text.

`tests/unit/threaded_diloco_test.py::test_broadcast_params_compiles_on_multi_device_cpu` starts a fresh subprocess with two virtual CPU devices. It compiles the extracted `make_broadcast_params_fn`, checks the leading `diloco` sharding and values, and inspects StableHLO. The subprocess sets the device-count flag before JAX initializes and cannot silently skip.

## Root cause

Three independent ownership/layout assumptions were invalid:

1. Broadcast transfer inputs had strict abstract `in_shardings`. This made the compiled input ABI require a device-local layout that a transferred/restored colocated-CPU buffer did not necessarily have. Output sharding is required; a frozen input ABI is not.
2. Fresh and restored syncer state was not uniformly canonicalized and completed on the current global CPU mesh before its first compiled consumer.
3. `pathwaysutils.concatenate_by_mesh_axis` explicitly always donates its arguments. `device_put` is allowed to alias when source and destination placement are equivalent, so CPU emulation deleted buffers still owned by learner state.

Checkpoint-resume testing found two further state-machine bugs:

4. With `pure_nnx=True`, generic checkpoint saving attempted `SyncerState.to_pure_dict()`, although the syncer is a Flax `struct.PyTreeNode`, not an NNX state.
5. After restoring at `start_step`, learners waited for the delayed response at `start_step`. Transport queues are ephemeral, so that pre-checkpoint in-flight message cannot exist after restart. The resume-time full-parameter broadcast already supersedes it.

CPU exposed one additional backend constraint: its learner and "colocated CPU" meshes are the same devices, and Pathways' donating cross-mesh primitive is not safe for overlapping meshes within one CPU PJRT client.

## Implemented fix

- `broadcast_params` and `broadcast_frag` retain exact `out_shardings` but no longer specify `in_shardings`. Their input placement and physical layout now come from the actual runtime buffer.
- Fresh and restored parameters plus outer optimizer state are placed with `may_alias=False` on the current global CPU shardings and blocked before broadcast compilation.
- The `cpu_clone` before fragment broadcast was removed; the producer JIT's correctly placed result flows directly into the layout-flexible broadcast.
- Learner-to-syncer transfers use `may_alias=False`, so downstream Pathways donation cannot invalidate learner state.
- CPU stack/unstack uses an explicit non-donating host-materialization fallback. TPU continues using Pathways concatenate/split, preserving the production communication path and overlap.
- The CPU integration test disables TensorBoard because it is unrelated to the algorithm and adds background writer processes/threads to an already threaded backend test.
- Checkpoint conversion now runs only for states that actually expose the NNX `to_pure_dict` protocol; native `SyncerState` is saved/restored through Orbax as its own PyTree.
- Delayed fragment receives now require `completed_step - tau > start_step`, dropping only messages from the lost pre-checkpoint overlap window.
- Each learner's first train-step compilation is serialized with the existing initialization lock. Subsequent steps remain concurrent, preserving steady-state overlap.

The lowered regression-test signature is now:

```text
func.func public @main(%arg0: tensor<8xf32>)
  -> tensor<2x8xf32> {sdy.sharding = ... "diloco" ...}
```

The input parameter has no frozen Sdy sharding/layout annotation, while the output still has the required `diloco` sharding. This is the intended transfer-boundary contract.

Do not add the `diloco` axis to learner model state as a workaround. That would put synchronization back into the learner SPMD program and undermine the intended overlap.

## Algorithm/design context

Streaming DiLoCo (arXiv:2501.18512v1, pp. 3–5) fragments synchronization, asynchronously sends one fragment, consumes the result after delay `tau`, and mixes it with the learner's meanwhile-advanced state. Real overlap requires the communication dependency to remain outside the learner's immediate compute path.

Decoupled DiLoCo (arXiv:2604.21428v1, pp. 3–9) uses learners and a CPU syncer as independent state machines connected by asynchronous messages. The syncer owns global parameters and outer optimizer state; learners own locally sharded training state. This supports the non-SPMD direction: only explicit communication payloads should acquire a learner/`diloco` dimension. Learner model tensors should not inherit the syncer's broadcast ABI.

## CPU test results (2026-07-20)

The documented MaxText 0.2.3 source environment was installed with Python 3.12, JAX/JAXlib 0.10.0, the TPU dependency set, and the pre-training GitHub dependencies. `uv pip check` passes after installing the six transitive dependencies omitted by the GitHub installer's `--no-deps` mode.

- Isolated two-device `broadcast_params` regression: **passed**.
- Complete `tests/unit/threaded_diloco_test.py`: **5 passed**.
- Four-device `test_threaded_diloco_minimal_run`: **passed**, including initial broadcast, two learners on disjoint two-device submeshes, repeated fragment synchronization, outer updates, broadcast, delayed receive, and mixing.
- Four-device `test_threaded_diloco_checkpoint_resume`: **passed**, including syncer/learner checkpoint save, process-local transport recreation, restore at step 2, canonical placement, full-parameter rebroadcast, skipped stale overlap receive, and continued training through step 4.
- StableHLO ABI assertion: **passed**; input is runtime-derived and output retains explicit `diloco` sharding.
- Threaded plus checkpoint unit suites: **14 passed**.
- Fresh plus resume integration suites: **2 passed**.
- Syntax compilation, `git diff --check`, and targeted undefined-name lint: **passed**.

The literal TPU `T(128)` error cannot be generated by CPU PJRT, so a final Pathways/TPU smoke test remains advisable. The CPU evidence nevertheless directly verifies the corrected ABI property that prevents the mismatch: the executable no longer requires an abstractly frozen physical input layout.
