# Bounded-memory non-SPMD streaming DiLoCo

Status: implemented and CPU-validated; TPU/Pathways execution validation
required

Date: 2026-07-23

## Summary

The previous non-SPMD streaming DiLoCo path could exhaust CPU memory with Qwen3-8B before
the apparent model size suggests that it should. This is not one isolated leak.
The current execution combines four multiplicative effects:

1. Qwen3-8B parameters use `float32` unless the launch overrides
   `weight_dtype`, so one parameter tree is about 30.51 GiB.
2. The CPU outer parameters and Nesterov momentum are placed on the global CPU
   mesh with no `diloco` entry in their `PartitionSpec`. JAX consequently
   replicates both trees on every DiLoCo slice.
3. Updating one scanned layer fragment uses a non-donated scatter into every
   full scanned parameter array and every full momentum array. A one-layer
   synchronization therefore creates about 51.75 GiB of new full-array data on
   each slice.
4. Every learner starts its default asynchronous step-zero AdamW checkpoint at
   the same time as the first outer synchronization. The logical checkpoint
   payload is about 91.54 GiB per learner.

For the checked-in two-slice Qwen3-8B launch, the state-only first-sync peak is
at least 112.78 GiB per CPU slice. If the step-zero learner checkpoint stages
the full logical state concurrently, the same slice can approach 204.32 GiB
before accounting for fragment averaging, transfers, executable temporaries,
or allocator fragmentation.

The design changes the memory complexity:

- the outer state lives on one coordinator CPU submesh, not once per learner;
- learner fragments are resharded and averaged one at a time;
- full scanned arrays are donated to their replacement scatter;
- transport reserves bounded capacity before allocating a CPU payload;
- initialization broadcasts are serialized and acknowledged;
- transfer and executable physical layouts are handled explicitly;
- large-model launches use BF16 weights unless FP32 is an intentional,
  capacity-checked choice; and
- checkpoint staging cannot race unconstrained with synchronization.

The target steady-state working set is `2P + O(F)` on one coordinator instead
of an aggregate `2NP` persistent state plus `O(NP)` full-array update peaks,
where `P` is a parameter tree, `F` is one fragment, and `N` is the number of
learners. The serialized full-tree initialization protocol has a separate
upper bound of about `3P + O(1)` on the coordinator while one destination copy
is in flight.

## Scope

This document covers the single-controller, non-SPMD implementation in:

- `src/maxtext/common/checkpointing.py`
- `src/maxtext/trainers/diloco/threaded_diloco.py`
- `src/maxtext/trainers/diloco/decomposed_transport.py`
- `src/maxtext/trainers/diloco/fragmenter.py`
- `src/maxtext/utils/mesh_utils.py`
- `scripts/diloco/run_streaming_diloco.sh`
- `scripts/diloco/run_streaming_diloco_gcloud.sh`
- `scripts/diloco/run_streaming_diloco_acceptance_workload.sh`

It covers CPU-resident communication payloads, outer state, averaging,
initialization, physical-layout adaptation, and checkpoint interaction.

It does not redesign the DiLoCo algorithm, the learner's TPU optimizer, or the
SPMD DiLoCo implementation. It also does not claim that CPU-only concurrent
PJRT execution faithfully emulates TPU Pathways execution.

## Workload used for the proof

Before this change, the checked-in streaming launch selected:

| Setting | Value |
| --- | ---: |
| Model | Qwen3-8B |
| DiLoCo slices | 2 |
| Decoder layers | 36 |
| Configured transformer fragments | 36 |
| Total fragments, including non-scanned fragment 0 | 37 |
| DiLoCo sync period | 36 |
| Communication overlap, `tau` | 2 |
| Launch steps | 20 |
| Weight dtype | inherited FP32 default |
| Learner optimizer | AdamW |
| Outer optimizer | Nesterov SGD |
| Checkpointing | enabled, asynchronous |
| Checkpoint period | 10,000 |
| Checkpoint staging concurrency | 96 GiB |

`FragmentedTreeManipulator` adds one non-scanned fragment, so the scheduling
interval is:

```text
max(1, round(diloco_sync_period / total_fragments))
= max(1, round(36 / 37))
= 1 training step
```

The first synchronization is therefore fragment 1, a scanned one-layer
fragment. The 20-step launch never reaches fragment 0 at step 37, so it is not
an adequate acceptance run for a complete streaming period.

## Quantified root-cause proof

### Qwen3-8B parameter size

The analytical count from the model configuration is 8,190,735,360
parameters:

```text
token embedding                         151,936 * 4,096
output projection                       4,096 * 151,936

per layer attention:
  q projection                          4,096 * 4,096
  k projection                          4,096 * 1,024
  v projection                          4,096 * 1,024
  output projection                     4,096 * 4,096

per layer SwiGLU MLP:
  two input projections                 2 * 4,096 * 12,288
  output projection                     12,288 * 4,096

plus layer, Q/K, and final normalization scales
```

An abstract pure-NNX Qwen3-8B model-tree audit reproduced that exact total:
6,946,071,552 scanned parameters and 1,244,663,808 non-scanned parameters.
It also confirmed that the concrete parameter scan axis is axis 1. Using
binary GiB:

| Symbol | Meaning | FP32 | BF16 |
| --- | --- | ---: | ---: |
| `P` | Full parameter tree | 30.51 GiB | 15.26 GiB |
| `S` | All 36 scanned decoder layers | 25.88 GiB | 12.94 GiB |
| `F_layer` | One scanned layer fragment | 0.719 GiB | 0.359 GiB |
| `F_0` | Non-scanned embedding/output fragment | 4.64 GiB | 2.32 GiB |

The old launch did not override `weight_dtype`, while
`src/maxtext/configs/base.yml` defaults it to `float32`. The CPU outer state
therefore receives FP32 parameters. Optax's Nesterov trace has the same shape
and dtype as the parameters, so the minimum full outer state is `2P`.

The repaired diagnostic launch now explicitly selects BF16 and disables
checkpointing. The table and proof above intentionally describe the failing
configuration.

### Global mesh replication

A `NamedSharding` consists of a `Mesh` and a `PartitionSpec`. If a mesh axis is
not used by a leaf's `PartitionSpec`, the value is replicated along that mesh
axis.

The old syncer creates its abstract and concrete state on `global_cpu_mesh`.
Normal model parameter specs do not contain `diloco`. Consequently every
parameter and every momentum shard is replicated across the `diloco` axis:

```text
persistent per slice       = params + momentum = 2P
persistent aggregate       = N * 2P

for N = 2 and FP32:
persistent per slice       = 61.03 GiB
persistent aggregate       = 122.05 GiB
```

This replication follows directly from JAX sharding semantics. It is not
removed by changing mesh contexts or by calling `jax.device_put` with the same
logical sharding.

### First scanned scatter

For a scanned fragment, `apply_flat_fragment` performs an indexed update on
every scanned leaf. Without donation, JAX must preserve the old input array
because Python is allowed to use it after the operation. Each indexed update
therefore returns a new full scanned leaf, even though only one layer changed.

The syncer applies this separately to:

- the outer parameter tree; and
- the Nesterov trace tree.

The state-only first-sync peak on every globally replicated slice is:

```text
old params                              P
old momentum                            P
new scanned parameter arrays            S
new scanned momentum arrays             S
                                      ----
minimum first-sync state peak       2P + 2S
```

For Qwen3-8B FP32:

```text
2P + 2S = 112.78 GiB per CPU slice
```

This is the decisive first-sync OOM mechanism. A 0.719 GiB fragment causes a
51.75 GiB full-array allocation because the persistent scanned representation
still contains all 36 layers.

### Live references make later peaks worse

The fresh initialization path retains local Python names for the incoming
learner parameters, the initial global parameters, and the initial optimizer
state. The restore path similarly retains the restore container. Learners keep
the received CPU initialization tree after copying it to TPU.

`gc.collect()` cannot release a live local or an aliased JAX array. After the
first sync, the original params and trace can remain pinned while the current
and next scanned states are created:

```text
pinned initial params and momentum      2P
current scanned params and momentum     2S
next scanned params and momentum        2S
                                      ----
later transient peak                2P + 4S
```

For Qwen3-8B FP32 this is 164.53 GiB per slice before fragment temporaries.
These lingering aliases also make donation unsafe or ineffective. Ownership
release is therefore a correctness requirement, not optional
garbage-collection tuning.

### Step-zero learner checkpoint collision

The learner calls checkpointing with Python loop step zero after its first
training step. Zero satisfies the default 10,000-step interval. Each learner
therefore begins an asynchronous checkpoint while the syncer is processing
fragment 1.

The default learner optimizer is AdamW. With FP32 weights, first moment, and
second moment, its logical persistent payload is approximately:

```text
params + mu + nu = 3P = 91.54 GiB per learner
```

Each learner has an independent checkpoint manager configured to permit 96 GiB
of concurrent checkpoint storage work. The exact resident staging peak is an
Orbax and Pathways implementation detail, but the configuration permits nearly
the full logical state to be in flight. It is unsafe to overlap that work with
the 112.78 GiB syncer peak.

### Controller-host materialization during stacking

The old stack implementation calls `np.asarray(shard.data)` for every
addressable CPU shard. In single-controller Pathways, remote colocated CPU
shards are addressable from the one Python client. NumPy conversion blocks and
fetches those shards into controller-host memory. The code then sends the data
back to a CPU device merely to introduce a leading size-one dimension.

It also retains:

- every original learner fragment;
- every expanded fragment copy; and
- the concatenated global fragment.

The resulting central-host and device working set grows with `N * F`. The
non-scanned FP32 fragment alone is 4.64 GiB per learner.

### Unbounded transport

The old transport has unbounded queues, unbounded out-of-order dictionaries,
and an unbounded `ThreadPoolExecutor` submission queue. More importantly, the
learner performs TPU-to-CPU `device_put` before submitting the background send.
A queue bound applied only to `Queue.put` would therefore be too late: the
large CPU allocation would already exist.

With the checked-in `tau=2`, normal receive blocking usually limits the
steady-state outstanding request count to approximately `tau + 1`. That makes
transport a secondary contributor to this particular first sync, not the
primary root cause. It remains an actual unbounded-memory bug under slower
syncers, larger overlap, resume mismatches, or failure.

## Before-and-after memory model

The table separates proven state storage from implementation-dependent
temporary buffers.

| Component | Before | Target |
| --- | --- | --- |
| Persistent outer state, aggregate | `2NP` | `2P` on coordinator |
| Persistent outer state, each learner slice | `2P` | zero outside coordinator |
| Scanned scatter state peak | `N(2P + 2S)` aggregate | `2P + O(F)` on coordinator |
| Learner averaging inputs | all `NF` retained | one incoming `F` at a time |
| Controller NumPy materialization | `O(NF)` | none |
| Pending transfer allocation | unbounded | reserved, fixed capacity |
| Initialization broadcast | potentially `O(NP)` in flight | one acknowledged target at a time |
| Concurrent learner checkpoints | up to `N * 3P` logical payload | serialized/capped |

For two learners:

| Case | FP32 | BF16 |
| --- | ---: | ---: |
| Old aggregate persistent outer state | 122.05 GiB | 61.03 GiB |
| Old first-sync state peak per slice | 112.78 GiB | 56.39 GiB |
| Coordinator-only persistent outer state | 61.03 GiB | 30.51 GiB |
| Donated coordinator state peak, excluding fragments | about 61.03 GiB | about 30.51 GiB |

The exact `O(F)` coefficient depends on XLA buffer assignment, resharding, and
Optax fusion. It must be measured on TPU. It is nevertheless bounded by a
small number of fragments rather than full models or learner count.

FP32 remains tight even after the architectural repair: the coordinator starts
at 61.03 GiB before fragments and runtime overhead. The Qwen3-8B launch should
therefore use `weight_dtype=bfloat16` unless FP32 outer-state capacity has been
measured and explicitly budgeted.

## Target architecture

### Mesh topology

The global TPU mesh retains the `diloco` axis only as an orchestration topology:

```text
global TPU mesh
  |
  +-- TPU learner submesh 0 -- colocated CPU submesh 0 (coordinator)
  +-- TPU learner submesh 1 -- colocated CPU submesh 1
  +-- ...
```

Each learner submesh removes the `diloco` dimension while preserving:

- device order;
- all remaining mesh axis names; and
- the corresponding `AxisType` values.

The coordinator is initially `cpu_submeshes[0]`. Its outer parameter and
optimizer shardings preserve each leaf's original model `PartitionSpec` and
memory kind, but use only the coordinator mesh.

No persistent outer leaf is committed to a mesh containing `diloco`.

### Coordinator-owned outer state

The coordinator owns exactly one logical copy of:

- outer parameters;
- Nesterov trace;
- outer step; and
- fragmenter metadata and executable caches.

The state remains sharded over model axes such as FSDP within the coordinator
submesh. Coordinator-only does not mean single-device or unsharded.

Checkpoint restore targets the coordinator shardings directly. It must not
restore onto a global CPU mesh and then select one replica.

### Streamed learner averaging

The syncer receives learner fragments in deterministic learner order. For each
learner:

1. receive one CPU-submesh fragment;
2. reshard it to the coordinator's corresponding fragment sharding;
3. donate the transport payload when ownership permits;
4. update a running sum or mean on the coordinator;
5. release the source before receiving another learner fragment.

Conceptually:

```python
running_sum = None
for learner_index in range(num_learners):
  source = recv_fragment(learner_index)
  local = reshard_to_coordinator(source, donate=True)
  running_sum = initialize_sum(local) if running_sum is None else add(running_sum, local)
mean = cast_to_param_dtype(running_sum / num_learners)
```

The sum update is compiled and donates both its accumulator and consumed
fragment. BF16 and FP16 inputs accumulate in FP32, then one final division is
cast back to the parameter dtype. Tests compare it with
`tree_map(lambda x: mean(x, axis=0), stacked)` at dtype-appropriate tolerances
because floating-point reduction order can differ.

This path replaces the production use of:

- `_expand_array_dims_with_mesh`;
- `np.asarray`;
- a global leading learner dimension; and
- `concatenate_by_mesh_axis`.

The stack helper can remain for focused compatibility tests, but it is not the
bounded-memory production averaging primitive.

### Outer update and donated scatter

For each fragment:

1. extract the coordinator's outer parameter fragment;
2. extract the corresponding Nesterov trace fragment;
3. compute `pseudo_gradient = outer - mean(inner)`;
4. apply the outer optimizer;
5. scatter the updated fragment into params and trace; and
6. replace the old `SyncerState` immediately.

Scanned full-array inputs are donated to the scatter executable. Before the
call:

- initialization aliases have been deleted;
- transport payloads cannot alias the state;
- no checkpoint operation owns the buffers; and
- no later computation references the old full arrays.

Donation allows XLA to alias the output to the full input buffer. It is an
optimization opportunity rather than an unconditional allocation guarantee,
so CPU tests inspect input deletion and TPU validation measures peak memory.
If Pathways cannot reuse the scatter inputs reliably, the fallback is to store
the outer state persistently as independently owned fragments and assemble a
checkpoint representation only at checkpoint boundaries.

Fragment extraction and scatter use layout-adapted JITs. Eager `jnp.take` and
eager `.at[].set` are not used on transferred or restored syncer arrays.

### Initialization protocol

Initialization is a high-risk peak because a full FP32 Qwen3-8B parameter tree
is 30.51 GiB.

The protocol is:

1. every learner advertises its restored step and the coordinator verifies
   that all checkpoint managers selected the same step;
2. learner 0 offloads one independently owned parameter payload;
3. the coordinator receives it and adopts or reshards it into coordinator
   ownership;
4. the coordinator initializes one Nesterov trace with the parameter inputs
   pinned to their existing physical formats, avoiding a model-sized layout
   conversion at this full-tree boundary;
5. all temporary source and wrapper references are deleted;
6. one independent destination copy is sent to one learner at a time;
7. that learner copies them to TPU, blocks until ready, deletes its CPU
   payload, and sends a small acknowledgment; and
8. only after the acknowledgment does the coordinator send to the next
   learner.

The learner-0 round trip may be skipped on a fresh run if the original TPU
state is provably the exact source used by the coordinator. Resume must always
install restored outer parameters.

A fragment-by-fragment initialization broadcast is a further optimization. It
reduces the largest destination payload from `P` to `max(F_0, F_layer)`, but it
is not required for the first bounded-memory implementation if the serialized
full-tree protocol fits.

### Updated-fragment broadcast

An updated fragment remains coordinator-owned while one independent transfer
payload is created for a learner CPU submesh. The transport reserves capacity
before that copy. The coordinator does not donate its source until every
required destination transfer has been initiated safely.

Because fragments are much smaller than `P`, each direction uses a fixed
capacity of `tau + 1` (with a minimum of one). This preserves the configured
communication overlap without permitting arbitrary growth.

### Bounded transport

Each learner and direction has a cancelable FIFO channel with a fixed capacity.
Its reservation covers unpublished transfers:

```text
reserve capacity
  -> allocate/launch device_put
  -> block in sender worker
  -> publish message
  -> receiver consumes message
  -> release capacity
```

Required behavior:

- reserve before `jax.device_put`;
- use `may_alias=False` for payloads that the receiver may donate;
- retain send futures and propagate their exceptions;
- reject out-of-order protocol messages rather than buffering without bound;
- cancel blocked reserve, publish, and receive operations on failure; and
- close all channels before waiting on sender workers.

A bounded `queue.Queue` alone is insufficient because it does not bound
transfers allocated before `put`.

### Checkpoint protocol

Checkpointing and donation impose an ownership conflict. A state buffer cannot
be donated to the next outer update while an asynchronous checkpoint still
depends on it.

The production protocol must:

1. checkpoint only at a consistent synchronization boundary;
2. prevent step-zero saves unless explicitly requested;
3. stop new donated updates while a snapshot still owns state buffers;
4. serialize learner checkpoint staging, or enforce a shared global byte
   budget instead of one 96 GiB budget per learner;
5. wait for initialization or outstanding transport ownership to settle;
6. checkpoint the coordinator state with coordinator shardings; and
7. start resume with empty transport channels and a fresh initialization
   broadcast.

Learner checkpoints now use the completed-step counter, matching the syncer's
state and checkpoint key. Periodic checkpoints must fall on fragment sync
steps, and a forced completion checkpoint is allowed only when the final step
is also a fragment sync step. The completion path forces that exact state step
through the shared checkpoint helper instead of relying on its legacy
loop-counter fallback. Continuous, emergency, automatic, and multi-tier
checkpoint modes are rejected by this non-SPMD path because they bypass the
single ownership lock. Invalid schedules and unsupported modes fail before
allocating model state. On resume, a learner ignores delayed response steps at
or before its restored `start_step`, since those queues intentionally restart
empty.

The shared lock serializes and bounds staging, but separate Orbax managers do
not form an atomic multi-directory transaction. A failure partway through a
checkpoint set can still leave different latest steps. The strict
initialization protocol rejects that mismatch instead of consuming an
out-of-order fragment; selecting the newest common checkpoint is future work.

For the 20-step diagnostic launch, checkpointing should be disabled. That is a
workload correction, not the complete production checkpoint solution.

For larger jobs, lower `checkpoint_storage_concurrent_gb` to a measured safe
value and prefer colocated Pathways checkpoint handling when supported.
Checkpoint success must be tested by save, process restart, restore, and
crossing at least one later synchronization boundary.

## JAX and Pathways API rationale

The implementation baseline audited here is JAX/JAXlib 0.10.0 and
`pathwaysutils` 0.1.8, matching this repository's generated TPU dependency
floors. The upstream `pathwaysutils` main branch (latest release 0.1.11 on the
document date) retains the same experimental `reshard` call contract. Because
that API and JAX's layout API are explicitly experimental, the deployed image
versions must still be recorded and revalidated.

### `Mesh`

`jax.sharding.Mesh` assigns ordered devices to named logical axes. Mesh order is
part of the sharding contract. Axis types also matter:

- `AxisType.Auto` allows automatic sharding behavior;
- `AxisType.Explicit` participates in explicit-sharding propagation; and
- other axis types must not silently become `Auto`.

Partitioning the global mesh must remove only the `diloco` axis and retain the
types of every surviving axis. Mesh and Flax axis-rule contexts are
thread-local; each learner establishes its own submesh context.

### `PartitionSpec` and `NamedSharding`

`PartitionSpec` maps array dimensions to mesh axes. A mesh axis absent from the
spec replicates the array along that axis. This is the direct reason that
placing ordinary model specs on the global CPU mesh creates `N` copies.

`NamedSharding(mesh, spec, memory_kind=...)` describes logical placement. It
does not by itself normalize device-local layout and it does not imply a copy.

### `Layout` and `Format`

Logical sharding and physical layout are separate:

- `Layout` describes device-local dimension order and tiling;
- `Format` combines a physical `Layout` with a `Sharding`.

TPU computation can produce tiled buffers while a TPU-to-CPU transfer or
checkpoint restore produces a different physical layout with the same shape,
dtype, and `NamedSharding`. A compiled executable checks its physical input
contract.

The executable is the authority. The layout adapter:

1. lowers and compiles the operation once;
2. reads `executable.input_formats`;
3. places each call's arguments into those exact formats; and
4. invokes the compiled executable.

This avoids hard-coding a TPU tile such as `T(256)` and avoids assuming that a
"null layout" is universal.

### `jax.device_put`

`jax.device_put` is asynchronous. A returned Python object does not imply that
transfer buffers can be discarded by a cooperating protocol.

Putting an already committed array to the same logical sharding can be a
metadata-only reuse. Therefore:

```python
jax.device_put(x, x.sharding)
```

does not establish a new physical layout and is not a layout normalizer.

Transport uses `may_alias=False` when the receiver may donate the payload.
Layout adaptation uses concrete `Format` targets, and donation is enabled only
when the old argument has a single owner.

### `jax.jit` donation

`donate_argnums` tells JAX that selected inputs will never be used after the
call. This permits input/output buffer aliasing for same-shaped scatter
results. Calling Python must treat donated arrays as invalid even if a backend
chooses not to reuse their buffers.

Donation is applied to:

- streamed averaging accumulators;
- consumed cross-mesh payloads where supported; and
- full scanned arrays during replacement scatter.

### Colocated CPU devices

`jax.experimental.colocated_python.colocated_cpu_devices(mesh)` creates a CPU
mesh corresponding to the TPU mesh. These are JAX devices and can be remote
from the central Python process. A CPU `jax.Array` is not equivalent to a local
NumPy array.

Calling `np.asarray` on such an array blocks and materializes it on the
controller host. The production path must keep values device-resident.

### Pathways resharding

Pathways resharding is the appropriate primitive for moving a committed
fragment between disjoint learner and coordinator CPU device sets. The target
is a tree of coordinator `NamedSharding`s preserving each source
`PartitionSpec` and memory kind. Resharding plans should be cached when the API
supports it.

The implementation uses:

```python
pathwaysutils.experimental.reshard.reshard(
    tree,
    target_sharding_tree,
    donate=True,
    may_alias=None,
    cache_resharding_plans=True,
)
```

The experimental side-channel API accepts logical `Sharding` trees, not
physical `Format` trees. It supports disjoint source and destination device
sets under a single-controller Pathways IFRT client. Startup verifies that all
colocated CPU submeshes came from the accelerator's client rather than the
fallback standalone CPU backend.

The API does not fast-path identity resharding. When every source leaf already
has exactly the requested coordinator sharding and ownership is being donated,
learner 0's tree is adopted directly and consumed by the donated reduction
instead of constructing a side-channel plan. The non-donating outgoing path
instead forces `jax.device_put(..., may_alias=False)` even for identical
shardings, so a queued learner-0 fragment cannot alias persistent coordinator
state.

The source may be donated only after transport ownership has transferred
completely to the syncer.

### `concatenate_by_mesh_axis`

`pathwaysutils.concatenate_by_mesh_axis` combines values from partitioned
meshes along a new mesh axis and donates its inputs. It is useful when the
algorithm requires a global leading dimension. Streaming mean does not require
that representation.

Avoiding it in the hot path:

- removes all-at-once input retention;
- removes the need to expand every fragment;
- avoids donor/alias hazards; and
- keeps outer computation on the coordinator mesh.

## Ownership and memory invariants

The implementation is correct only while all of these invariants hold:

1. Exactly one CPU submesh owns persistent outer params and momentum.
2. No persistent outer sharding contains the `diloco` mesh axis.
3. A donated array has no live Python alias and no asynchronous consumer.
4. TPU-to-CPU transport payloads do not alias live learner parameters.
5. Capacity is reserved before allocating a transport payload.
6. At most the configured number of unpublished or queued messages exists per
   learner and direction.
7. Messages are monotonically ordered by the protocol; unexpected messages
   fail loudly.
8. Production synchronization never converts remote CPU arrays to NumPy.
9. Cross-mesh input is released before the next learner input is accepted,
   apart from the explicitly bounded transport slot.
10. Executable input `Format`, not a guessed layout, defines physical layout.
11. Mesh slicing preserves device order, axis names, and axis types.
12. Initialization sends at most one full destination payload at a time.
13. Checkpointing cannot overlap donation of the same buffers.
14. Resume begins with empty queues; pre-checkpoint overlap messages are not
    expected after restart.

These invariants should appear as assertions, protocol errors, tests, or
comments at the ownership boundary rather than remaining implicit.

## Failure handling

If any learner, sender worker, or syncer fails:

1. record and re-raise the first exception;
2. close the shared transport manager;
3. wake blocked producers and consumers;
4. cancel unused reservations;
5. shut down sender executors; and
6. wait for already-started device work only when it cannot deadlock shutdown.

Without cancellation, a capacity-one queue can turn a memory fix into a
shutdown deadlock when the syncer exits while a learner worker is blocked.
An asynchronous offload failure closes the complete protocol from the worker
itself, before the learner reaches its next future poll, because the missing
fragment cannot be recovered by merely returning queue capacity.

## CPU evidence and limitations

CPU tests can establish:

- the Qwen3-8B byte calculations from shape/dtype metadata;
- omitted mesh-axis replication semantics;
- submesh axis-type preservation;
- bounded reservation before payload allocation;
- strict FIFO and cancellation behavior;
- propagation of background-send failures;
- post-resume delayed-step filtering and checkpoint-schedule validation;
- independent same-mesh outgoing payloads;
- streamed-mean numerical equivalence on small PyTrees;
- lack of `np.asarray` in the production averaging path;
- donation eligibility and `jax.Array.is_deleted()` after scanned scatter;
- format-adapter behavior with constructed CPU layouts.

The final focused CPU suite completed with 40 passing tests and two environment
skips. It covers the streamed reducer, donated scanned scatter, bounded and
cancelable transport, strict FIFO, background error propagation, mesh axis
types, fragment-level outer updates, the DiLoCo data loader, and the layout
adapter, including explicit completed-step checkpoint handling. A forced
two-device CPU execution of the normally Pathways-gated layout regression
passed as well.

A separate abstract Qwen3 pure-NNX check verified that, after removing the
`diloco` axis, every BF16 parameter and Nesterov-trace leaf has a non-null
sharding on the coordinator mesh. This caught and fixed a restore-specific
gap: the old pure-NNX abstract optimizer state did not carry sharding metadata.

CPU tests cannot establish:

- TPU physical tiling chosen by XLA;
- real colocated sidecar placement or memory accounting;
- Pathways reshard peak memory and plan reuse;
- whether TPU buffer assignment realizes every requested donation;
- Orbax staging behavior on the target topology; or
- stability of fully concurrent learner and syncer execution on TPU.

Past fully concurrent virtual-CPU runs have encountered native PJRT
instability. Focused CPU tests remain useful evidence, but a CPU native crash is
not by itself proof of a TPU bug or fix.

## TPU acceptance plan

The runnable launcher for these phases is:

```text
scripts/diloco/run_streaming_diloco_gcloud.sh
```

It defaults to the `mlperf-v5p` cluster and `v5p-8` slices, but requires the
project, cluster location, and output bucket to be supplied by the caller.
Start with a no-side-effect rendering of the launch:

```bash
PROJECT_ID=PROJECT \
LOCATION=LOCATION \
BASE_OUTPUT_DIRECTORY=gs://BUCKET/maxtext \
scripts/diloco/run_streaming_diloco_gcloud.sh plan tiny
```

Then submit `layout`, `tiny`, and `qwen8b` in that order. The launcher packages
only the source tree, the live layout test, and its container runner; it does
not send local virtual environments or unrelated workspace files in the Docker
build context. It passes TPU compiler flags through XPK's Pathways proxy-server
argument, verifies colocated CPUs use the TPU IFRT client, disables XPlane for
the baseline, samples controller cgroup/RSS memory, and prints commands for
observing every Pathways pod and container.

The optional `tiny-save` followed by `tiny-resume` phases exercise the
checkpoint step-alignment handshake without paying for an 8B checkpoint.

A recommended sequence is:

```bash
export PROJECT_ID=PROJECT
export LOCATION=LOCATION
export BASE_OUTPUT_DIRECTORY=gs://BUCKET/maxtext
export IMAGE="gcr.io/${PROJECT_ID}/maxtext-diloco-acceptance:streaming-fix"

scripts/diloco/run_streaming_diloco_gcloud.sh plan tiny
scripts/diloco/run_streaming_diloco_gcloud.sh submit layout
SKIP_BUILD=1 scripts/diloco/run_streaming_diloco_gcloud.sh submit tiny
SKIP_BUILD=1 scripts/diloco/run_streaming_diloco_gcloud.sh submit qwen8b
```

The Qwen phase deliberately starts at `per_device_batch_size=1` and
`max_target_length=512`. This isolates coordinator CPU-state memory from TPU
activation pressure. After it passes, exercise the original workload shape
without changing the acceptance-defining model, precision, fragment schedule,
or step count:

```bash
SKIP_BUILD=1 scripts/diloco/run_streaming_diloco_gcloud.sh submit qwen8b \
  per_device_batch_size=8 max_target_length=2048
```

Only those two MaxText overrides are accepted. The live container reports its
baked source revision, installed JAX/JAXLIB/pathwaysutils versions, accelerator
and colocated-CPU devices, and the installed Pathways reshard signature before
training. It fails rather than silently falling back if the backend, slice
count, colocated-client relationship, or reshard API is incompatible. The
launcher also selects the requested GKE context before any `kubectl` operation,
waits for complete log capture before applying pass criteria, and bounds
preflight and phase runtime so a transport deadlock cannot consume the TPU
reservation indefinitely.

### Phase 1: layout and transfer smoke test

Use two small learner submeshes and a tiny model.

1. Transfer fresh learner fragments to colocated CPU.
2. Reshard each fragment to the coordinator.
3. Run extract, mean, outer update, donated scatter, and response transfer.
4. Repeat with a checkpoint-restored coordinator state.
5. Verify no `INVALID_ARGUMENT` physical-layout mismatch.
6. Verify every response is committed to the intended learner CPU submesh
   before its TPU transfer.

### Phase 2: Qwen3-8B bounded-memory run

Use the checked-in two-slice topology with:

- `weight_dtype=bfloat16`;
- checkpointing disabled;
- 36 transformer fragments;
- `tau=2`; and
- at least 80 steps.

Eighty steps crosses fragment 0 twice and exercises more than two complete
37-fragment periods. Twenty steps is insufficient.

Collect:

- controller process RSS and cgroup memory;
- each colocated CPU worker/sidecar RSS and cgroup memory;
- OOM-kill and allocator logs;
- fragment ID and message depth;
- initialization acknowledgment timing;
- reshard plan creation/reuse;
- compile count per fragment signature; and
- synchronization latency.

Acceptance criteria:

- no CPU or TPU OOM;
- memory reaches a bounded plateau after warmup;
- no period-over-period full-model growth;
- no controller-host spike proportional to `N * F`;
- transport depth never exceeds its configured bound;
- all 37 fragment IDs complete in order;
- the non-scanned fragment completes successfully; and
- no layout or deleted-buffer error occurs.

### Phase 3: correctness comparison

On a model small enough to run both methods:

1. run the former stacked mean as a reference;
2. run the streamed coordinator mean with identical inputs;
3. compare averaged fragments, pseudo-gradients, outer parameters, and momentum
   after every fragment;
4. use exact comparison for integer/control values and dtype-appropriate
   tolerance for floating-point reductions; and
5. cover both `communication_overlapping_alpha=0` and a nonzero value.

### Phase 4: checkpoint and resume

1. Enable a deliberately small, measured checkpoint concurrency budget.
2. Save after at least one complete synchronization period.
3. Wait for checkpoint completion.
4. terminate the job;
5. restore the coordinator and learner states;
6. run through the next full period; and
7. verify that no learner waits for a pre-restart transport message.

Memory must remain bounded while saving and after restoration. The restored
arrays must pass the same physical-format adapter as fresh transfers.

### Phase 5: scaling and failure injection

Repeat with more learners and verify:

- coordinator persistent memory is independent of learner count;
- only communication volume and bounded fragment slots scale with learners;
- a deliberately stalled learner applies backpressure before another CPU
  payload allocation;
- a syncer exception wakes blocked learners;
- a learner sender exception reaches the orchestrator; and
- shutdown completes without a blocked executor.

### Optional FP32 stress test

FP32 Qwen3-8B should be attempted only after BF16 acceptance and only on a
coordinator with a documented budget above:

```text
61.03 GiB persistent outer state
+ largest fragment working set
+ Pathways/XLA allocator reserve
+ checkpoint reserve, if enabled
+ process and executable overhead
```

Passing BF16 does not imply that FP32 has adequate headroom.

## Risks and mitigations

### Donation is not guaranteed to eliminate every copy

Mitigation: verify deletion in CPU tests, inspect TPU buffer-assignment evidence
where available, and measure actual peak memory. Use fragmented persistent
state if full-array donation is not realized.

### Cross-mesh resharding may introduce a new transient

Mitigation: process one learner at a time, donate consumed sources, cache plans,
and measure coordinator and source sidecar memory independently.

### Running mean changes floating-point reduction order

The implementation already uses a streamed sum followed by one division.
Low-precision inputs accumulate in FP32, which is more stable but is not
bitwise identical to a BF16 stacked reduction.

Mitigation: compare against the stacked reference at realistic dtypes and
document tolerances.

### Coordinator-only initialization can recreate an `NP` peak

Mitigation: serialize with acknowledgments or broadcast fragments instead of a
full tree. A bounded queue without consumption acknowledgment does not prove
that the destination CPU payload has been released.

### Asynchronous checkpoints can invalidate donation assumptions

Mitigation: coordinate snapshot ownership explicitly and wait for staging or
save completion before donating the same state. Use a shared staging budget.

### Strict FIFO can expose resume protocol bugs

This is intentional. Silently buffering unexpected messages hides stale or
missing synchronization steps and defeats the memory bound.

Mitigation: the learner calculates the first post-resume response explicitly,
rejects steps at or before the restored state, and starts with empty channels.
Checkpoint schedules are aligned to synchronization boundaries; a partially
written multi-manager checkpoint set is still detected rather than repaired.

### JAX and Pathways layout APIs are version-sensitive

Mitigation: keep the supported JAX and `pathwaysutils` versions pinned in the
runtime image, derive formats from compiled executables, and run the layout
smoke test for every image update.

### BF16 changes optimizer numerics

Mitigation: treat `weight_dtype=bfloat16` as an explicit workload decision,
validate loss and outer-update behavior, and retain a separately budgeted FP32
mode when required.

## Alternatives rejected

### Keep global state and only reduce queue size

This leaves the 61.03 GiB FP32 state on every slice and the 112.78 GiB scanned
scatter peak. It cannot fix the root cause.

### Coordinator-only state without donated scatter

This reduces aggregate replication but leaves the coordinator's 112.78 GiB
first-sync state peak. It is necessary but insufficient.

### Convert all fragments through NumPy

This centralizes remote data, blocks overlap, and scales controller memory with
learner count. It is useful only as a small CPU-test reference.

### Normalize layouts with same-sharding `device_put`

`jax.device_put(x, x.sharding)` may return or alias `x`; a `NamedSharding` does
not specify physical tiling. It does not satisfy an executable `Format`
contract.

### Bound only `Queue.put`

The learner allocates the CPU payload before enqueueing it. The allocation
would remain unbounded in the executor work queue.

### Rely on `gc.collect()`

Garbage collection cannot release live locals, transport aliases,
asynchronous checkpoint owners, or runtime work. Ownership must be explicit.

### Disable checkpointing as the complete fix

Disabling the diagnostic step-zero checkpoint removes a major concurrent peak,
but global replication and non-donated full scanned scatters independently
exceed a safe memory budget.

## Implementation map

The implementation now covers these responsibilities:

| File | Responsibility |
| --- | --- |
| `checkpointing.py` | explicit forced saves at an exact completed-state key |
| `threaded_diloco.py` | coordinator mesh, streamed mean, initialization acknowledgments, ownership release, checkpoint coordination |
| `decomposed_transport.py` | pre-allocation reservation, bounded FIFO, cancellation, future error propagation |
| `fragmenter.py` | format-adapted extraction and donated full-array scatter |
| `mesh_utils.py` | axis-type-preserving partitioning and layout-safe device operations without host NumPy |
| `run_streaming_diloco.sh` | BF16 diagnostic launch and intentionally disabled checkpoints |
| `run_streaming_diloco_gcloud.sh` | reproducible `mlperf-v5p` image build, XPK submission, and cluster observation |
| `run_streaming_diloco_acceptance_workload.sh` | Pathways API preflight, staged acceptance configs, bounded runtime/memory sampling, and exact pass criteria |
| unit tests | transport bounds/failure, donation, streamed mean, layout, mesh types, and outer-update correctness |

The checked-in profiler launch intentionally remains 20 steps. It is useful
for confirming that the former first-sync OOM is gone, but it does not reach
fragment 0. The TPU acceptance run must override it to at least 80 steps as
described above.

## Decision

Adopt coordinator-only outer state, streamed averaging, explicit ownership, and
donated scatter as one coherent change. None of those items should be removed
in isolation to simplify the patch: each addresses a different term in the
proven peak.

Use BF16 and disable checkpointing for the first Qwen3-8B TPU acceptance run.
Re-enable checkpointing only with a bounded, coordinated staging protocol.

The change is accepted only after a real two-slice TPU/Pathways run completes
multiple full fragment periods, including fragment 0, with a stable CPU-memory
plateau.

## Primary API references

- [JAX distributed arrays and explicit sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)
- [JAX device-local layout control](https://docs.jax.dev/en/latest/notebooks/layout.html)
- [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html)
- [JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html)
- [Google Cloud: Port JAX workloads to Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/porting-jax-workloads)
- [Google Cloud: Run a Pathways batch workload](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/batch-workload)
- [Google Cloud: Troubleshoot Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/troubleshooting-pathways)
- [`pathwaysutils.experimental.reshard` source](https://github.com/AI-Hypercomputer/pathways-utils/blob/main/pathwaysutils/experimental/reshard.py)
- [`pathwaysutils.experimental.concatenate_by_mesh_axis` source](https://github.com/AI-Hypercomputer/pathways-utils/blob/main/pathwaysutils/experimental/concatenate_by_mesh_axis.py)
