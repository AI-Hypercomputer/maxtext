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

# RL Trainer Benchmark

## Overview

Tools for standalone AOT compilation, HBM estimation, and TPU benchmarking of
the RL trainer, without needing rollout engines
or an orchestrator:

| Tool | Where it runs | When to use it |
| :--- | :--- | :--- |
| `run_trainer_compile.sh` | **CPU only** (no TPU slice needed) | Use before requesting TPUs to verify that the trainer compiles and fits in device HBM. |
| `render_trainer_jobset.py` | **CPU only** (no TPU slice needed) | Use to convert an e2e launch script into a standalone GKE `JobSet` YAML with `run_trainer.py` embedded. |
| `run_trainer.py` | **Trainer TPU slice** (Pathways client pod) | Runs the trainer benchmark (`fwd_bwd` and `update` steps) on TPUs to measure step time and capture profiler traces. |

---

## End-to-End Workflow

### Step 1: Export a `DRY_RUN=true` manifest from Tunix

From a checkout of the [`google/tunix`](https://github.com/google/tunix)
repository, run the distributed RL recipe launcher with `DRY_RUN=true` so it
prints the rendered Kubernetes `JobSet` manifests without submitting them to the cluster:

```bash
DRY_RUN=true \
  TRAINER_BACKEND=maxtext \
  TRAINER_JOBSET_YAML=jobset.pathways.yaml \
  TUNIX_IMAGE=<tunix-docker-image> \
  MAXTEXT_CKPT=gs://<bucket>/ckpt/0/items \
  bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher.sh --command start > launch-dryrun.yaml
```

### Step 2: Check compilation and HBM peak on CPU

Before requesting a TPU slice, verify that your trainer configuration compiles
and check its per-device HBM peak on CPU:

```bash
LAUNCH_RENDER=launch-dryrun.yaml \
  bash src/maxtext/experimental/maxtext_engine/rl/run_trainer_compile.sh
```

`run_trainer_compile.sh` extracts `MAXTEXT_EXTRA_FLAGS`, `LIBTPU_INIT_ARGS`,
the model and sequence parameters, and the mesh/topology from
`launch-dryrun.yaml` (via `render_trainer_jobset.py --print-aot-env`) and runs
`maxtext.training_engine.maxtext_engine_compile`, printing the per-kernel memory
breakdown and `TRAIN-KERNEL DEVICE PEAK`. You can pass variant overrides via
`TRAINER_SLICE`, `EXTRA_MAXTEXT_FLAGS`, `REPLACE_MAXTEXT_FLAGS`,
`LIBTPU_EXTRA`, `LIBTPU_DROP`, or `COMPUTE_LOGPS_CHUNK_SIZE`.

### Step 3: Run the trainer benchmark on a TPU slice

Once the configuration passes the AOT check, there are two ways to run RL trainer benchmarks depending on your environment:

#### Scenario A: Run on GKE via `JobSet`

1. **Render the standalone trainer `JobSet` YAML**:
   Run `render_trainer_jobset.py` on your workstation. It extracts
   `TRAINER_ARGV`, `ORCH_ARGV`, `MAXTEXT_EXTRA_FLAGS`, and `LIBTPU_INIT_ARGS`
   from `launch-dryrun.yaml`, embeds `run_trainer.py` into the `proc`
   container's entrypoint (with `--maxtext_ckpt_path=""` for random
   initialization), and writes `rendered/<job-name>.yaml` (or a custom path if
   `--out` is passed):

   ```bash
   python3 src/maxtext/experimental/maxtext_engine/rl/render_trainer_jobset.py \
     --launch-render launch-dryrun.yaml \
     --name <job-name> \
     --namespace <namespace> \
     --output-prefix gs://<bucket>/benchmark
   ```

2. **Apply the rendered YAML on GKE**:
   `kubectl apply` launches the trainer `JobSet` on the TPU slice and runs the
   embedded `run_trainer.py` automatically inside the `proc` pod:

   ```bash
   kubectl --context <context> -n <namespace> apply -f rendered/<job-name>.yaml
   kubectl --context <context> -n <namespace> logs -f -c main \
     -l jobset.sigs.k8s.io/jobset-name=<job-name>,jobset.sigs.k8s.io/replicatedjob-name=proc 2>&1 | tee -a <job-name>.log
   kubectl --context <context> -n <namespace> delete jobset <job-name>
   ```

   > **Note:** Run `python3 src/maxtext/experimental/maxtext_engine/rl/render_trainer_jobset.py --help` to see optional override, slice, and profiler flags (such as `--steps`, `--micro-steps`, `--trainer-slice`, `--profiler-steps`, and `--extra-maxtext-flags`).

#### Scenario B: Interactive TPU / Pathways session

If you already have an active TPU or Pathways session, you can skip
`render_trainer_jobset.py` and run `run_trainer.py` directly by passing
`TRAINER_ARGV` and `ORCH_ARGV` in the environment:

```bash
STEPS=3 MICRO_STEPS=4 PAYLOAD=realistic \
  TRAINER_ARGV="--worker_id=trainer-bench --trainer_backend=maxtext ..." \
  ORCH_ARGV="--model_id=Qwen/Qwen3.5-397B-A17B ..." \
  python3 -m maxtext.experimental.maxtext_engine.rl.run_trainer
```
