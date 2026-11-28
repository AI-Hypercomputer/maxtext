<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# MPMD pipeline parallelism in pure JAX (experimental)

MPMD pipeline parallelism for mixture-of-experts decoders (DeepSeek-V3; the Qwen3.5 family, Qwen3.8-2.4T-A95B
included) in plain JAX, with no MPMD partitioner: each stage is its own set of jitted `shard_map` programs on its
own sub-mesh (FSDP x EP, the FSDP collectives folded into the microbatch programs), a stage boundary is a
device-to-device copy, and one Python thread replays a fixed schedule without blocking. It runs on TPU in one
process, as multi-controller JAX across hosts and slices (`--distributed`), or as one Pathways client
(`JAX_PLATFORMS=proxy`), and on NVIDIA GPUs (`--platform gpu`, one process a host).

**Status: experimental.** A training harness with its own command line (`ds_ep_pipeline --help`), not a MaxText
trainer: synthetic data (its own, or MaxText's with `--data maxtext`) or a token file (`--data npy:PATH`), a constant
learning rate, no checkpointing. Its layer math is a pure-JAX port of each model that the tests hold to MaxText's; its
kernels are MaxText's (megablox, the ring of experts, Lineage's layers for DeepSeek-V3's batch split) and tokamax's
splash attention. Run its modules as `python3 -m maxtext.experimental.mpmd.jax.<module>`, never by file path (the
package is named `jax`).

## Environment setup

| component | version | notes |
|---|---|---|
| MaxText | `c034bc02f` (main, 2026-11-25) plus this package | from source, `.[tpu]` |
| Python | 3.12 | |
| jax / jaxlib | 0.12.0.1 | |
| libtpu | 0.0.51 | MaxText's lowest; the benchmarks below ran on 0.0.50.1 (jax 0.12.0.1's own pin), and the TPU-topology (AOT) tests have the same outcomes on both |
| tokamax | 0.0.18 | the benchmarks' version, pinned below (MaxText's lowest, 0.0.17, is what its CI runs: the CPU tests pass on both) |
| pathwaysutils | 0.1.14 | Pathways only |
| Pathways server, proxy | `20261124-jax_0.12.0.1` (GKE), `20261117-jax_0.12.0.1` (local) | the image's jax must be the client's; plain `jax.jit`, which this is, also runs on builds after 2026-11-17 (Shardy MPMD does not) |
| typeguard | 2.13.3 | only for DeepSeek-V3's batch split on Lineage's layers (not in MaxText's requirements; 4.6.1 fails to import Lineage) |
| triton, jax-triton | 3.9.0, 0.5.0 | only for the GPU Triton kernels |

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git && cd maxtext   # a checkout with this package
uv venv --python 3.12 --seed venv && source venv/bin/activate
uv pip install -e .[tpu] tokamax==0.0.18 --resolution=lowest && install_tpu_pre_train_extra_deps
uv pip install typeguard==2.13.3   # optional: DeepSeek-V3's batch split on Lineage's layers
```

Commands run from the repository root. The launchers (`ds_ep_run.sh`, `ds_ep_multihost.sh`, `ds_ep_aot.sh`) build
`LIBTPU_INIT_ARGS` from `libtpu_flags/$MPMD_FLAG_SET.flags` (`tpu7x`, the default: 30 flags, among them the 64 MiB
scoped VMEM limit the megablox tiles need; `v5p`; `none`) plus `--xla_max_concurrent_async_collective_permutes=64`
and `EXTRA_LIBTPU_INIT_ARGS`; `LIBTPU_INIT_ARGS_OVERRIDE` replaces the set.

### Pathways

One client process drives every TPU through the IFRT proxy, so the pipeline takes its one-process path: every stage
is local and each boundary is one IFRT `CopyArrays`, across slices too; `--layout slices` puts one stage on each
slice, or k as strips of whole chips at k times the slices (two stages a TPU7x-4x4x4 are its z-halves). The
Pathways servers compile, so the libtpu flags (`python3 -m maxtext.experimental.mpmd.jax.pathways flags tpu7x`
prints the set's 31 on one line) go to the proxy, the resource manager and the workers, and the same set to the
client's `LIBTPU_INIT_ARGS`, which the planner reads.

The multi-controller code (one process a host, `--distributed`) runs only where `jax.process_count() > 1`
(`pathways.multi_controller()`), so never under Pathways, whose one client is the only process. (A one-host run skips it
too, but for what it asks for: `--distributed` still starts `jax.distributed`, and `ds_ep_aot` still checks a multi-host
target's hosts, unless `--pathways` plans one Pathways client's run.) Under Pathways there is no walk over other
processes' boundary copies, no key-value agreement or barrier, no relay, xfer or cross-process copy, no per-process
record or trace. Its flags are refused there before the backend starts, each naming what to use instead:
`--distributed`, `--transport relay` and `xfer`, `--json-procs` and `--profile-procs` other than `leaders` (Pathways
traces `--profile-hosts` worker hosts), and `--norm-reduce allreduce`; `ds_ep_multihost.sh` starts nothing. Its
environment variables are not refused, since a cluster may set them for every job. `--max-ahead` defaults to 2 there,
since nothing else bounds how far the client runs ahead; the wait for finished steps runs beside the dispatch, so the
bound does not slow it.

**In docker on one TPU VM** (as a GKE Pathways job's pods run them; a TPU7x-8 is `tpu7x:2x2x1`, a v6e-4
`tpuv6e:2x2`). One client may hold the proxy at a time, and the worker restarts after each one disconnects:

```bash
IMG=us-docker.pkg.dev/cloud-tpu-v2-images/pathways TAG=20261117-jax_0.12.0.1 GCS=gs://<bucket>/pathways-scratch
FLAGS=$(python3 -m maxtext.experimental.mpmd.jax.pathways flags tpu7x)   # unquoted below: one argument a flag
sudo docker run -d --name pw-rm --net=host -e TPU_SKIP_MDS_QUERY=true -e JOBSET_NAME=pw \
  -e REPLICATED_JOB_NAME=pathways-head -e HOST_ADDRESS=127.0.0.1 $IMG/server:$TAG --server_port=29001 \
  --gcs_scratch_location=$GCS --node_type=resource_manager --instance_count=1 --instance_type=tpu7x:2x2x1 $FLAGS
sudo docker run -d --name pw-worker --net=host --privileged --restart=on-failure -e XCLOUD_ENVIRONMENT=GCP \
  -e MEGASCALE_GRPC_ENABLE_XOR_TRACER=false -e MEGASCALE_NUM_SLICES=1 -e MEGASCALE_SLICE_ID=0 -e JOBSET_NAME=pw \
  -e REPLICATED_JOB_NAME=worker -e PATHWAYS_HEAD=127.0.0.1 -e MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1 \
  $IMG/server:$TAG --server_port=29005 --resource_manager_address=127.0.0.1:29001 --gcs_scratch_location=$GCS $FLAGS
sudo docker run -d --name pw-proxy --net=host -e PATHWAYS_HEAD=127.0.0.1 $IMG/proxy_server:$TAG \
  --server_port=29000 --resource_manager_address=127.0.0.1:29001 --gcs_scratch_location=$GCS $FLAGS
# ready when the resource manager logs "1/1 Pathways Slices Now Ready" and the proxy "IFRT proxy server started"
sudo docker rm -f pw-rm pw-worker pw-proxy   # when done: frees the chips
```

**On GKE**, follow [MaxText's Pathways guide](../../../../../docs/run_maxtext/run_maxtext_via_pathways.md) (Cluster
Toolkit, an image holding this tree, `gcluster job submit --pathways`), with the flag set on the three Pathways
containers (`"$FLAGS"`, quoted: one argument an option) and their images at the client's jax. The run under
[Benchmark results](#benchmark-results) went through xpk 1.16.0 as the last command does; the `gcluster` one has not
been submitted, and gcluster v1.108.0 and v1.109.0 rerun a finished Pathways job and end it Failed (cluster-toolkit
PR #6739, merged to its develop branch on 2026-11-25): use a gcluster built with that fix, or xpk. Expect a cold
compile of 25-30 minutes with no client output (Pathways then caches the programs under `--pathways-gcs-location`);
`--profile` takes only a `gs://` directory.

```bash
PW=us-docker.pkg.dev/cloud-tpu-v2-images/pathways TAG=20261124-jax_0.12.0.1
FLAGS="$(python3 -m maxtext.experimental.mpmd.jax.pathways flags tpu7x)"
CMD="JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 MPMD_PATHWAYS_IMAGE=$TAG LIBTPU_INIT_ARGS='$FLAGS' \
  python3 -m maxtext.experimental.mpmd.jax.ds_ep_pipeline --layout slices --stages 2 --vv 2 --fsdp 16 --ep 8 \
  --layers 8 --experts 256 --vocab 129280 --gbs 1024 --mb 8 --embed gather --acc-dtype bfloat16 --remat-last \
  --ragged-buffer-factor 2 --expert-weights sliced --bench --steps 20 --json gs://<bucket>/run.json"
gcluster job submit --pathways --name=${RUN_NAME?} --image=${DOCKER_IMAGE?} --compute-type=tpu7x-standard-4t \
  --topology=4x4x4 --num-slices=2 --pathways-gcs-location=gs://<bucket>/pathways-scratch --command="$CMD" \
  --pathways-server-image=$PW/server:$TAG --pathways-server-args="$FLAGS" --pathways-worker-image=$PW/server:$TAG \
  --pathways-worker-args="$FLAGS" --pathways-proxy-server-image=$PW/proxy_server:$TAG --pathways-proxy-args="$FLAGS"
xpk workload create-pathways --cluster=${CLUSTER?} --workload=${RUN_NAME?} --docker-image=${DOCKER_IMAGE?} \
  --tpu-type=tpu7x-4x4x4 --num-slices=2 --pathways-gcs-location=gs://<bucket>/pathways-scratch --command="$CMD" \
  --server-image=$PW/server:$TAG --custom-pathways-server-args="$FLAGS" --custom-pathways-worker-args="$FLAGS" \
  --proxy-server-image=$PW/proxy_server:$TAG --custom-pathways-proxy-server-args="$FLAGS"
```

## Quick start

```bash
# 1. CPU dry run: build the pipeline and print its plan, compiling nothing (seconds)
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  bash src/maxtext/experimental/mpmd/jax/ds_ep_run.sh --stages 2 --vv 2 --fsdp 1 --ep 2 --layers 4 --experts 16 \
  --gbs 16 --mb 8 --dry-run

# 2. One TPU7x-8 (McJAX): 4 MoE layers at DeepSeek-V3's per-layer dimensions, 128 experts, PP=2 x FSDP 2 x EP 2
bash src/maxtext/experimental/mpmd/jax/ds_ep_run.sh --bench

# 3. Pathways, the local proxy above: two stages on two fake slices of one TPU7x-8
TAG=20261117-jax_0.12.0.1 FLAGS="$(python3 -m maxtext.experimental.mpmd.jax.pathways flags tpu7x)"
JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 MPMD_PATHWAYS_IMAGE=$TAG LIBTPU_INIT_ARGS="$FLAGS" \
  MPMD_FAKE_SLICES=2 python3 -m maxtext.experimental.mpmd.jax.ds_ep_pipeline --layout slices --stages 2 --vv 2 \
  --fsdp 2 --ep 2 --layers 4 --experts 16 --vocab 129280 --gbs 16 --mb 4 --embed gather --acc-dtype bfloat16 \
  --remat-last --ragged-buffer-factor 2 --bench --steps 20 --json gs://<bucket>/run.json

# 4. Multi-controller JAX on every host of a 2 x TPU7x-4x4x4 job: across slices the relay (or ppermute), as jax's
#    native cross-slice copy hangs
RUN_DIR=gs://<bucket>/run bash src/maxtext/experimental/mpmd/jax/ds_ep_multihost.sh --layout slices --stages 2 \
  --vv 2 --fsdp 16 --ep 8 --layers 8 --experts 256 --vocab 129280 --gbs 1024 --mb 8 --embed gather \
  --acc-dtype bfloat16 --remat-last --ragged-buffer-factor 2 --expert-weights sliced --transport relay --bench
# --model qwen3.8-2.4t-a95b switches to that model's dimensions; --layers 61 --dense-layers 3 --mlp 18432
# --experts 256 --vocab 129280 is the whole DeepSeek-V3
```

## Zero-bubble schedules (ZB-H1 / ZB-H2)

`--schedule zbh1` / `zbh2` (at `--vv 1`) run the zero-bubble schedules of Qi et al.: each middle chunk's backward is
cut into I, the input gradient with the recompute it needs, and W, the weight gradients, which run later, in the
bubbles. The cut is the chunk's own backward program split by data dependence (`stage_ad.split_jaxpr`), so no model
code changes. Chunk 0's backward stays whole, and so does the fused last chunk unless `--zb-split-last on` (off by
default) cuts it too: its F runs the forward, the head, the loss and the input gradient (AI), its W its layers' weight
gradients. Turn it on at two stages, where nothing else splits (without it zbh1 is 1F1B's order). On one TPU7x-8 it
was faster in all 22 pairs measured (0.7-3.3%, mean 1.5%), but at G4 below slower: zbh2 3448.8 ms against 3332.4
without it (+3.5%). ZB-H1 keeps 1F1B's residual sets and ZB-H2 about twice them, and each split stage holds up to
`--zb-wbuf-cap` (default 1; 0 for no cap) pending W buffers on top: a microbatch's crossing set and what its W reads,
7.25 GiB a device for two DeepSeek-V3 MoE layers at 4096 tokens a device (a run's is in the summary's `split backward`
line). Each schedule is the fastest of its candidates in simulation under the program costs (`--costs`); the run warns
where its order simulates slower than 1F1B's.

```bash
# L4 below, zbh2 on one TPU7x-8 (McJAX); through the local Pathways proxy, the same flags with quick start 3's env
bash src/maxtext/experimental/mpmd/jax/ds_ep_run.sh --stages 4 --vv 1 --fsdp 1 --ep 2 --layers 8 --experts 64 \
  --gbs 16 --mb 8 --schedule zbh2 --zb-split-last on --bench   # --zb-wbuf-cap 2: two W buffers a stage
```

**Correctness.** On CPU (XLA:CPU's YNNPACK fusions off) zbh1 and zbh2 give 1F1B's losses, gradients and updates bit
for bit: the halves run the fused programs' operations, and each chunk's W's add its gradients in microbatch order.
MaxText's CPU CI runs with the fusions on, which compile some halves otherwise: there those pipeline checks skip, the
three-stage fp32 ones hold zbh1 / zbh2 to a few ulps, and the split checks allow fp32 leaves 4 ulps (its runners gave
fp32 experts' gradients on XLA's ops up to 2 ulps apart, fusions off). On a TPU they do not, as XLA:TPU compiles
each half by itself. The tests hold it to a contract measured on a TPU7x-8 (McJAX and local Pathways): every leaf
of a split chunk within 3% of the fused program's (bit for bit with excess precision off, but MLA's `q_norm` /
`kv_norm` scales); warmup losses within 2e-3 and gradient norms within 2e-2 of 1F1B's (seen: 4.0e-4 and 6.8e-3 on
the tests' TINY, 3.4e-5 and 8.2e-4 at the widths below); zbh2 zbh1's bit for bit.

**Results**, ms a step (run-ahead mean of 20; against 1F1B on the same backend). DeepSeek-V3 per-layer widths,
megablox, bf16, one 4096-token row a device a microbatch:
- L4 `--stages 4 --vv 1 --fsdp 1 --ep 2 --layers 8 --experts 64`, L2 `--stages 2 --vv 1 --fsdp 2 --ep 2 --layers 4`,
  Qwen: L4 at Qwen3.8-2.4T-A95B's widths with 32 experts; one TPU7x-8 (2026-11-27; jax 0.12.0.1, libtpu 0.0.50.1),
  McJAX on this tree, local Pathways (image `20261117-jax_0.12.0.1`) on it before the McJAX pacing below, which
  Pathways does not use, and before Pathways' default of at most 2 steps ahead (this tree's L4 at M=8 came within
  0.1%).
- G4 and G2: 2 x TPU7x-4x4x4 on GKE, Pathways image `20261124-jax_0.12.0.1`, the widths of the GKE run under
  [Benchmark results](#benchmark-results) with `--layout slices --gbs 1024`; G4 `--stages 4 --vv 1 --fsdp 8 --ep 8
  --mb 16` (2026-11-28; one job, evicted twice: 1F1B ran in its first client session, breadth and zbh2 with the split
  in its second, the rest in its third; zbh2 without it repeats an earlier job's 3338.6 within 0.2%), G2
  `--stages 2 --vv 1 --fsdp 16 --ep 8 --mb 8` (2026-11-27).
- `--vv 2` breadth: the fastest schedule without the split on the TPU7x-8 (G2's: the GKE run under Benchmark results).
- split last: `--zb-split-last` in the zero-bubble columns (G2 ran before it existed).

| run | M | split last | 1F1B | zbh1 | zbh2 | zbh2 `--zb-wbuf-cap 2` | `--vv 2` breadth |
|---|---|---|---|---|---|---|---|
| L4, McJAX | 8 | on | 1337.5 | 1262.0 (−5.6%) | 1230.6 (−8.0%) | 1205.4 (−9.9%) | 1157.3 (−13.5%) |
| L4, Pathways | 8 | on | 1441.2 | 1369.9 (−4.9%) | 1332.9 (−7.5%) | 1306.7 (−9.3%) | 1254.9 (−12.9%) |
| L4, McJAX | 16 | on | 2329.1 | 2248.4 (−3.5%) | 2216.7 (−4.8%) | 2200.3 (−5.5%) | 2118.2 (−9.1%) |
| L4, Pathways | 16 | on | 2501.4 | 2444.7 (−2.3%) | 2405.9 (−3.8%) | 2379.6 (−4.9%) | |
| L2, McJAX | 8 | on | 1388.3 | 1342.4 (−3.3%) | 1302.2 (−6.2%) | | 1332.2 (−4.0%) |
| L2, Pathways | 8 | on | 1459.4 | 1410.8 (−3.3%) | 1367.2 (−6.3%) | 1367.6 (−6.3%) | 1422.5 (−2.5%) |
| Qwen, McJAX | 8 | on | 2492.3 | 2340.6 (−6.1%) | 2228.4 (−10.6%) | | |
| Qwen, Pathways | 8 | on | 2569.1 | 2419.4 (−5.8%) | 2368.2 (−7.8%) | | |
| G4, GKE | 16 | on | 3621.7 | 3477.9 (−4.0%) | 3448.8 (−4.8%) | 3422.4 (−5.5%) | 3626.9 (+0.1%) |
| G4, GKE | 16 | off | | | 3332.4 (−8.0%) | | |
| G2, GKE | 8 | off | 3227.6 | 3257.6 (+0.9%: 1F1B's order) | | | 2943.2 (−8.8%) |

On the TPU7x-8 at four stages `--vv 2` breadth stays ahead of every zero-bubble order; at two, zbh2 passes it. At G4
breadth gains nothing over 1F1B, and zbh2 without `--zb-split-last` is the fastest run measured there. The split last
chunk puts W buffers on the last stage too (L4 McJAX: its peak 66.1 GiB, 51.7 without it, 79.9 under 1F1B; ahead of
time G4 41.1 GiB on its last stage, 33.7 without it).
Measured (`xprof costs`), a split chunk's I + W is 3.6-4.5% above its B at DeepSeek-V3 widths and the split last
chunk's AI + W 4.5-6.3% above its A (Qwen: within 1%), so with enough microbatches the split's overhead outgrows the
bubble it fills.

**On a TPU VM's own client (McJAX)** a program's outputs are allocated when the one dispatch thread issues it, and the
loop runs ahead of the chips. Unpaced, the W buffers of I's issued early filled the split stages' HBM whatever
`--zb-wbuf-cap` said (87.5 of 94.7 GiB at L4), and a split stage's last W sent libtpu into defragmenting, holding
every stage's dispatch 0.6-0.7 s a step: L4's zbh1 / zbh2 ran about 2% / 4.5% under 1F1B, against the 4.0% / 6.3%
they simulate. So the loop issues an I (and a split last chunk's AI) only once its chunk's previous one has
finished, and each stage's trailing W's last; Pathways, whose client issues asynchronously, is not paced. Dispatch is
still one thread: where a stage's own working set leaves its HBM short, its waits hold up every stage, so leave the
split stages headroom (fewer layers a stage, shorter microbatches) or run on Pathways.

## Benchmark results

Step times are the mean of run-ahead steps after warm-up; TFLOP/s per device by MaxText's accounting. This port's
numbers below were measured on MaxText `4b5b24c92`; on `c034bc02f` quick start 2 gives the same losses bit for bit
and the same step time (1333 ms).

**Pathways against multi-controller JAX, 2 x TPU7x-4x4x4 on GKE** (2026-11-26). 8 DeepSeek-V3 MoE layers (256
experts top-8, random routing), vocab 129280, seq 4096, GBS 1024 in 8 microbatches; PP=2 with one stage a slice,
FSDP 16 x EP 8, V=2 circular, bf16 weights / accumulation / reduce-scatters; 93.9B parameters (the GKE command
above). Pathways: this port, image `20261124-jax_0.12.0.1` with the 31 `tpu7x` flags (client jax 0.12.0.1, libtpu
0.0.50.1, tokamax 0.0.18). McJAX: the pre-port implementation (unpublished), 2026-10-22, one process a host with the
host relay (jax 0.11.1, libtpu 0.0.46, tokamax 0.0.12).

| | Pathways, `--transport native` | McJAX, `--transport relay` (pre-port) |
|---|---|---|
| step, run-ahead mean of 20 | **2943.2 ms** (2.9% longer), 209.8 TFLOP/s/device | **2859 ms**, 216.0 TFLOP/s/device |
| step, blocked median of 20 | 2947.2 ms | 2861 ms |
| a boundary crossing, 58.7 MB a device | 41-51 ms (isolated copy: 44.6 slice 0 to 1, 40.7 back, 51.4 both at once) | 50 ms (trace median) |
| warm-up losses | 13.2037582 / 13.1034050 / 13.0033779 | 13.2037678 / 13.1033659 / 13.0034189 |
| compile; init | 40 programs in 509 s, ~25 cached (cold ~25-30 min, at the head pod's 8-thread compile service); 114.8 s | 19 programs (stage 1) in 99 s; 52.4 s |

**Cross-slice copies under Pathways, 2 x TPU7x-4x4x4** (image `20261124-jax_0.12.0.1`, 2026-11-25 and 26): one client
moved 128 devices' shards to the other slice exactly, device to device, and programs on the two slices overlapped. A
small crossing in a dependent chain took ~1.5 ms (0.26 ms of it dispatch); 1 GiB (8 MiB a device) 14.7 ms blocked and
139 GB/s back to back by the batched copy (`device_put`: 17.4 ms, 119 GB/s); the run above's 7.5 GB boundary
40.7-44.6 ms blocked (median), 211-254 GB/s back to back; 1 GiB inside one slice (64 to 64 devices) 344-381 GB/s back
to back.

**One TPU7x-8**, this port (2026-11-26; jax 0.12.0.1, libtpu 0.0.50.1, tokamax 0.0.18): quick start 2 (McJAX) ran at
1334 ms a step, 214.3 TFLOP/s/device, peak HBM 77.0-79.5 GiB (23.95B parameters; 40 programs compiled in 161 s); quick
start 3 (local Pathways, image `20261117-jax_0.12.0.1`, flags on the servers) at 734 ms, 240.9 TFLOP/s/device, and
with `--expert-weights sliced` at 732 ms native against 1145 ms with `--transport ppermute`, the same losses. On the
tests' TINY pipelines (precompiled, `--bench --steps 20`; 2026-11-25, the pre-port implementation, whose programs this
port reproduces bit for bit) a Pathways step (image `20261124`) took 10.03 ms against McJAX's 11.75 at S=2 (two fake
slices of FSDP 2 x EP 2) and 21.20 against 27.62 ms at S=4 (FSDP 2). Such steps are dispatch-bound (a call costs
~80-140 us through the proxy whatever its devices, ~50 us a device on McJAX), so at S=8 Pathways trailed, 75-100
against 49-53 ms a step (measured on the code before its Pathways mode).

**Numerics.** Pathways does not reproduce McJAX bit for bit: the client lowers the same programs (every hash
matches), but the servers compile them with their own XLA and libtpu, a few ulps apart. Over TINY's three warm-up
steps the losses differ by up to 5.5e-4 relative at S=2 and 2.0e-3 at S=4 (gradient norms 1.6e-2 and 2.8e-2); at the
GKE size above, against the McJAX relay run, losses 3.2e-6 and gradient norms 3.3e-5. Each image repeats itself bit
for bit, `native` and `ppermute` boundaries alike.

**Earlier results**, from the pre-port implementation (unpublished):

| result | hardware | date |
|---|---|---|
| Against MaxText's SPMD pipeline, 4 DeepSeek-V3 MoE layers, PP=2 x FSDP 2 x EP 2: 1603 against 2668 ms a step (1.66x; 1.52x at equal FLOPs) | 1 x TPU7x-8 | 2026-09 |
| The whole DeepSeek-V3 (671B), PP=8 one stage a slice, against MaxText's DP over DCN: 1.43x at GBS 1024 (8515 against 12150 ms), 1.17x at 2048 | 8 x TPU7x-4x4x4 | 2026-10-25 |
| With 1024 devices a slice the order reverses: MaxText's DP over DCN beat PP=4 over DCN on the whole DeepSeek-V3 (6746 against 8076 ms, 1.20x) and Qwen3.8-2.4T-A95B (20069 against 26499 ms, 1.32x), one 4096-token sequence a device | 4 x TPU7x-8x8x8 | 2026-11-17 |
| Qwen3.8-2.4T-A95B, PP=8: all 92 layers fit (AOT, 56.6-78.8 GiB of 94.74); 8 layers trained at 5054 ms a step | 8 x TPU7x-4x4x4 | 2026-10-25 |
| Batch split, interleaved 1F1B, PP over RDMA: whole DeepSeek-V3 2374 tokens/s/device (1.19x DP on the same nodes), whole Qwen3.8-2.4T 869 | 64 / 188 GB200 | 2026-11-09 to 11 |

## Tests

The tests are in `tests/unit/experimental/mpmd/jax/` and use MaxText's markers only. MaxText's `tests/conftest.py`
marks every test without `tpu_only` / `gpu_only` as `cpu_only`, skipped on accelerators: on a TPU VM run those with
`JAX_PLATFORMS=cpu`. A test that needs more devices than jax has skips, naming the count.

```bash
# What MaxText's cpu-unit job runs on this pull request (4 fake CPU devices, scheduled_only tests included)
XLA_FLAGS=--xla_force_host_platform_device_count=4 python3 -m pytest -n 8 \
  -m 'cpu_only and not post_training and not integration_test' tests/unit/experimental/mpmd/jax
# The fast tier on 8 devices, 4-7 minutes on a 32-core machine
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 python3 -m pytest -n 8 --dist loadfile \
  -m 'not scheduled_only and not tpu_only and not gpu_only' tests/unit/experimental/mpmd/jax
# By hand, what no CI job runs on CPU: the 8-device tests, the slow checks (those alone ~2 h of test time) and the
# tests that start several JAX processes (multi-controller JAX over local CPU processes)
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 MAXTEXT_MPMD_SLOW_TESTS=1 \
  MAXTEXT_MPMD_MULTIHOST_TESTS=1 \
  python3 -m pytest -n 8 -m 'not tpu_only and not gpu_only' tests/unit/experimental/mpmd/jax
# On TPUs, a TPU VM's own client; on Pathways JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 instead
# (a pytest process of its own: one client per proxy)
JAX_PLATFORMS=tpu MAXTEXT_MPMD_REQUIRE_TPU=1 \
  python3 -m pytest tests/unit/experimental/mpmd/jax/pathways_test.py -k OnTpu
```

| MaxText CI job | what runs here |
|---|---|
| cpu-unit: 4 CPU devices, libtpu installed, `JAX_PLATFORMS` unset | every test but the `tpu_only` and `gpu_only` ones; `scheduled_only` ones nightly, and on a pull request that adds or changes them (this one: all 1190, of which 934 pass and 256 skip, in 24-29 minutes with 8 workers on a replica of its 32-core runner; the tests that start several JAX processes skip unless `MAXTEXT_MPMD_MULTIHOST_TESTS=1`, as they need more memory than a CI worker running other test modules has) |
| tpu-unit (v6e-4); tpu7x-unit (tpu7x-8, on a pull request only the tests it touches); TPU Pathways unit (v6e-4 through the proxy) | the 18 `tpu_only` tests: `pathways_test.OnTpu` runs (one slice of 4n devices: two stages of FSDP n x EP 2, and a skip on a job of several slices; its four-stage zero-bubble test needs 8), its two-stage split-last test and `SplitBackwardOnTpu` (`scheduled_only`) nightly; the 12 Lineage checks skip, as MaxText's TPU images have no typeguard (with typeguard 2.13.3, `MAXTEXT_MPMD_REQUIRE_LINEAGE=1 JAX_PLATFORMS=tpu python3 -m pytest -m tpu_only tests/unit/experimental/mpmd/jax/ds_ep_lineage_test.py tests/unit/experimental/mpmd/jax/ds_ep_lineage_pipeline_test.py` runs them on a TPU7x-8, over an hour; some are McJAX-only and skip on Pathways) |
| gpu-unit (A100) | the 13 `gpu_only` kernel checks; those needing Triton / jax-triton or an sm_100 GPU skip |

## Known limitations

- Multi-controller JAX across TPU slices: jax's native copy hangs and `jax.experimental.transfer`'s receiver crashes
  (jax 0.11.1, libtpu 0.0.46): use `--transport relay` or `ppermute` (the run warns), or Pathways. Across v6e or v5p
  hosts the native receive aborts.
- Pathways: one client per proxy, no peak HBM, no host callbacks, last-bit differences from McJAX, slow cold compiles.
- On a TPU VM's own client, libtpu keeps a sub-mesh's collectives on the TensorCore when it holds chip (0,0,0) but
  not the whole slice (Pathways offloads them): `--layout slices` or `physical_origin_last` avoids it.
- MaxText's SparseCore unsort caps EP x tokens a device x top-k (Qwen3.8 at 4096 tokens a device: EP 16, not 32);
  `--prefetch-pin auto` is calibrated on v5p and v6e only.
- The zero-bubble schedules: one chunk a stage only; not on Lineage's batch split, with stash modes or under
  `--fsdp-scope repeat`; on a TPU VM's own client one dispatch thread paces every stage (see above).
- Not run since the port: multi-host multi-controller jobs, the `gcluster` Pathways command, `OnTpu`'s 4-device
  form (MaxText CI's v6e-4), and the zero-bubble schedules on any TPU but a TPU7x.
