---
name: maxtext-release
description: Orchestrates the full MaxText PyPI release validation and testing lifecycle, ensuring strict regression checking and detailed reporting.
---

# MaxText Automated Release Preparation Skill

This skill guides the AI agent in performing end-to-end validation and preparation for a MaxText release. MaxText releases require high confidence; safety and correctness are prioritized over speed.

## Critical Execution Rules

1. **Strict NO-GO Default (Zero Regression & Zero Skip Tolerance)**: The default executive recommendation must **always** be **NO-GO**. You must **never** declare a "GO" recommendation if any testing track timed out, aborted early, failed, or was marked as **SKIPPED** (even if you determine a skip or failure was due to infrastructure issues rather than code regressions). A "GO" recommendation is strictly reserved for runs where 100% of the verification suites execute to absolute completion and return pristine exit statuses. **Not releasing at all is better than pushing out a bad release.**
2. **Halt and Preserve Forensic State on Failure**: If any test, compilation, or workload fails, you must **immediately halt execution** and report verbatim stderr diagnostics to the maintainer. **CRITICAL**: Do **not** automatically trigger cleanup routines or delete virtual environments, output directories, or cluster workloads after a failure! You must preserve the broken environment exactly as it is so human maintainers can inspect core dumps, pod logs, and tensorboard traces. Do not attempt automatic code fixes.
3. **User Ownership**: The user is responsible for the final release decision. Provide a detailed, organized report at the end of the process sufficient for them to make the final call.
4. **Mandatory Clean State on Start or Explicit Reset**: Always initiate a brand new verification run from a clean baseline. If the maintainer explicitly directs a restart or clean reset, invoke the exhaustive cleanup instructions in **Phase 0B** to terminate orphaned processes, clear active cluster queues, and purge transient caches. Never invoke Phase 0B automatically upon encountering a test failure.
5. **Execution Duration Awareness**: Be advised that critical steps (such as compiling container images or executing distributed XPK workloads) can take **up to an hour** to finalize. Ensure command execution tools are invoked asynchronously or configured with sufficient timeout allowances (exceeding standard 5-minute execution limits) to prevent abrupt process termination.
6. **Verbatim Output Preservation**: The executing agent must capture and preserve every command string and its raw, verbatim stdout/stderr outputs in full. To prevent context bloat, write these extensive logs to separate local scratch files or persistent artifacts, and present explicit clickable file links to them within the final summary report.
7. **Proactive Storage Capacity Monitoring**: Local artifacts (such as cached Hugging Face tokenizers/models, multi-gigabyte base Docker layer extractions, and localized staging wheels) quickly consume host disk capacity. Actively monitor storage usage (`df -h .`) before executing storage-heavy compilation or caching workflows. If available space drops below a safe threshold (e.g., 20GB free capacity), alert the maintainer immediately to coordinate cache purges (`docker system prune`, cache purges).
8. **Strict Sequential Cluster Scheduling (Zero Concurrency)**: Distributed multi-host GKE clusters have finite hardware slice capacity (e.g., 128 or 256 chips). You must submit distributed XPK verification workloads **strictly sequentially**. Never submit multiple XPK workloads in parallel. Before submitting any workload, query the active cluster queue (`xpk workload list`) to confirm zero conflicting jobs are running. You must actively monitor each submitted workload to 100% completion and verify its clean exit status before launching the next workload in the suite.

## Required Runtime Information (Ask User)

Before initiating the workflows, ask the user to provide the following variables if not already established in the context:
- **Target Release Version**: The version string to validate and release (e.g., `0.2.3`).
- **Hugging Face Token (`HF_TOKEN`)**: Required for accessing gated models during inference and training tests.
- **Base Output Directory**: GCS path or local directory for test outputs (`BASE_OUTPUT_DIRECTORY`).
- **Single-Host Target**: The hostname or IP of the TPU VM to use for single-host tests. *(Note: Execute the context parsing decision tree in **Phase 0 Item 2** first. If you are already running directly on a dedicated TPU VM, skip prompting for this variable and default to `localhost`)*.
- **Multi-Host Cluster Configuration**:
  - GKE Cluster Name (`CLUSTER_NAME`)
  - Google Cloud Zone (`ZONE`)
  - Google Cloud Project ID (`PROJECT`)
  - TPU Type for XPK workloads (`TPU_TYPE`, e.g., `v6e-256`)

## Phase 0: Pre-Flight Environment Sanity Checks

Before executing builds or launching long-running jobs, verify that the foundational infrastructure is active and accessible. If any check fails, halt execution and report the explicit blocker to the user.

### 1. Host Storage, Daemon Access, & Registry Upload Readiness
Verify available storage capacity, ensure working sudoless Docker access, and perform a lightweight container push test to confirm write permissions to your configured image repository (e.g. `gcr.io/$PROJECT` or custom registry):
```bash
# 1. Storage and daemon socket check
df -h .
docker --version
docker ps

# 2. Container registry upload authentication smoke test
export TEST_REPO="gcr.io/$PROJECT/maxtext-auth-probe-$(date +%s)"
docker pull hello-world
docker tag hello-world $TEST_REPO
docker push $TEST_REPO
docker rmi $TEST_REPO hello-world
```
- **Disk Space Check**: Verify the host partition has at least **20GB of available space**. If space is critical, halt and request permission to run cache pruning routines (`docker system prune -a -f`, clear `~/.cache/huggingface/`).
- **Docker Daemon Check**: If `docker ps` returns `permission denied`, guide the user to configure daemon socket permissions (`sudo usermod -aG docker $USER`) and refresh their session context.
- **Registry Upload Check**: If `docker push` fails with `unauthorized` or `denied`, guide the user to run `gcloud auth configure-docker` or verify IAM permissions on the target repository before proceeding.

### 2. Target TPU Hardware Health & Single-Host Execution Context
Execute a multi-stage decision tree to parse your local vs. remote execution environment and verify low-level JAX accelerator hardware health:
1. **Check Active Shell Context**: Run `python3 -c "import jax" 2>/dev/null`. If this succeeds without error, your active shell is already inside a JAX virtual environment. Proceed immediately to run the hardware health check below.
2. **Check Standard Host Virtual Environments**: If the active shell lacks JAX, search for existing standard virtual environment paths on the local filesystem (e.g., `~/maxtext_venv/`, `~/venv/`, or `./venv/`). If found, source it (`source <path>/bin/activate`) and re-test `import jax`.
3. **Ask User for Confirmation**: If no JAX virtual environment is found on the host, prompt the user: *"Are you currently executing directly on a dedicated TPU VM?"*
4. **Local TPU Provisioning (If User says YES)**: If confirmed, create a baseline TPU virtual environment locally following official repository installation standards:
   ```bash
   uv venv ~/maxtext_preflight_venv --python 3.12
   source ~/maxtext_preflight_venv/bin/activate
   uv pip install maxtext[tpu] --resolution=lowest
   ```
   Once installed, execute the hardware health verification check below.
5. **Remote SSH Execution (If User says NO)**: If the user confirms they are running on a remote runner machine without local accelerators, prompt for the remote target SSH details (Hostname/IP, Project ID, Zone, SSH User) and execute all Single-Host TPU verification commands remotely over SSH (`gcloud compute tpus tpu-vm ssh ...`).

**Hardware Health Verification Command**:
```bash
# Must be executed inside an active JAX virtual environment
python3 -c 'import jax, jax.extend ; print(f"{jax.__version__=}\n{jax.devices()=}\n{jax.extend.backend.get_backend().platform_version=}")'
```
- Confirm the command outputs valid TPU device mappings and backend strings. If JAX initialization fails (e.g., `No devices found` or `RuntimeError`), flag immediately as a critical blocker.

### 3. GKE Cluster Verification
Confirm the existence and running state of the target multi-host cluster pool:
```bash
gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE --project=$PROJECT
```
- Ensure the cluster configuration resolves successfully and returns a `RUNNING` status.

---

## Phase 0B: Abort Recovery & Restart Cleanup Routines

**CRITICAL NOTE ON FORENSIC PRESERVATION**: Never execute these cleanup routines automatically after a test failure! If a failure occurs, halt and preserve all state for debugging. Only execute these cleanup routines when initiating a brand new validation run or when the maintainer explicitly instructs a clean reset.

When directed to reset or abort, execute the following exhaustive cleanup steps to restore the infrastructure to an absolute baseline:

### 1. Terminate Orphaned Processes & Workloads
- **Remote Single-Host JAX Processes**: Kill any lingering Python training threads on the TPU VM to release accelerator device locks (`JaxRuntimeError: ABORTED: The TPU is already in use`):
  ```bash
  sudo pkill -9 -f "python3 -m maxtext"
  ```
- **Active Distributed Cluster Workloads**: Query and forcefully purge any pending or running background verification jobs from the GKE clusters:
  ```bash
  xpk workload list --cluster $CLUSTER_NAME --project=$PROJECT --zone=$ZONE
  # For any active workloads associated with this testing run:
  xpk workload delete --workload <workload_name> --cluster $CLUSTER_NAME --project=$PROJECT --zone=$ZONE
  ```

### 2. Purge Transient Storage, Virtual Environments, & Docker Artifacts
Completely strip local cache directories, built wheel distributions, isolated virtual environments, remote cloud storage buckets, and transient container images associated with the aborted release:
```bash
# Purge transient Python virtual environments
rm -rf ~/maxtext_pre_* ~/maxtext_post_* ~/maxtext_runner_* ~/maxtext_bench_runner_*

# Clean local compilation output directories and root staging packages
rm -rf dist/ build/ *.whl

# Remove remote staged PyPI wheel distributions from test GCS bucket
gcloud storage rm -r gs://maxtext_wheels/test/$VERSION/

# Purge persistent training outputs and checkpoint structures saved during the aborted run
if [[ "$BASE_OUTPUT_DIRECTORY" == gs://* ]]; then
  gcloud storage rm -r "$BASE_OUTPUT_DIRECTORY"
else
  rm -rf "$BASE_OUTPUT_DIRECTORY"
fi

# Purge local Docker container layers and images built during this execution run
docker rmi -f maxtext_base_image maxtext_base_image__runner 2>/dev/null || true
docker system prune -f # Clear dangling intermediate build cache layers
```

### 3. Reset Git Source Directory
Ensure the local repository working tree contains zero leftover execution configs or untracked build overrides:
```bash
git reset --hard HEAD
git clean -fdx # Force clean untracked execution artifacts and leftover test overrides
```

---

## Phase 1: Verification & Packaging

### 1. Version Check
- Inspect `src/maxtext/__init__.py` to verify that `__version__` matches the **Target Release Version**.
- If it does not match, alert the user and pause until they update it or authorize you to proceed.

### 2. Build PyPI Distribution
Execute locally on the runner machine:
```bash
# Ensure clean state
git status # Verify no uncommitted changes

# Set target version
export VERSION=<Target Release Version>

# Build wheel using uv
uv build --wheel

# Verify wheel exists in dist/
ls -l dist/

# Upload to staging GCS bucket for testing
gcloud storage cp dist/* gs://maxtext_wheels/test/$VERSION/
```

## Phase 2: Single-Host TPU Validation

Execute these steps on the designated **Single-Host Target** TPU VM. Ensure a fresh virtual environment is used for each sub-phase.

### Sub-phase 2A: Pre-Training Verification
Execute on TPU VM:
```bash
export VERSION=<Target Release Version>
export BASE_OUTPUT_DIRECTORY=<User Output Directory>
export HF_TOKEN=<User HF Token>

# Clean environment setup
uv venv --python 3.12 --seed ~/maxtext_pre_$VERSION
source ~/maxtext_pre_$VERSION/bin/activate

# Fetch staging wheel
gcloud storage cp gs://maxtext_wheels/test/$VERSION/maxtext-$VERSION-py3-none-any.whl .

# Install package and extra dependencies
uv pip install maxtext-$VERSION-py3-none-any.whl[tpu] --resolution=lowest
install_tpu_pre_train_extra_deps
```

**Tests to Run Sequentially**:
1. **Llama2-7b Pre-training (No Checkpoint)**:
   ```bash
   export RUN_NAME=pre-no-ckpt-$(date +%Y%m%d%H%M%S)
   python3 -m maxtext.trainers.pre_train.train run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY dataset_type=synthetic steps=5 model_name=llama2-7b
   ```
2. **Llama2-7b Pre-training (With Checkpoint)**:
   ```bash
   export RUN_NAME=pre-ckpt-$(date +%Y%m%d%H%M%S)
   python3 -m maxtext.trainers.pre_train.train run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY dataset_type=synthetic steps=5 model_name=llama2-7b load_parameters_path=gs://maxtext-model-checkpoints/llama2-7b/2025-01-23-19-26/scanned/0/items
   ```
3. **Inference Decoding**:
   ```bash
   python3 -m maxtext.inference.decode model_name=llama2-7b tokenizer_path=meta-llama/Llama-2-7b tokenizer_type=huggingface hf_access_token=$HF_TOKEN scan_layers=false per_device_batch_size=1 ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 max_prefill_predict_length=128 max_target_length=256 prompt="I love to" attention=dot_product
   ```

### Sub-phase 2B: Post-Training Verification
Execute on TPU VM in a **new** virtual environment:
```bash
export VERSION=<Target Release Version>
export BASE_OUTPUT_DIRECTORY=<User Output Directory>
export HF_TOKEN=<User HF Token>

uv venv --python 3.12 --seed ~/maxtext_post_$VERSION
source ~/maxtext_post_$VERSION/bin/activate

# Fetch staging wheel if not present
gcloud storage cp gs://maxtext_wheels/test/$VERSION/maxtext-$VERSION-py3-none-any.whl .

# Install post-training package
uv pip install maxtext-$VERSION-py3-none-any.whl[tpu-post-train] --resolution=lowest
install_tpu_post_train_extra_deps

# Ensure HF authentication is configured for the environment
# (Agent should guide user if manual login is needed)
```

**Tests to Run Sequentially**:
1. **SFT on Llama3.1-8b-Instruct**:
   ```bash
   export RUN_NAME=sft-single-$(date +%Y%m%d%H%M%S)
   export MODEL=llama3.1-8b-Instruct
   export MAXTEXT_CKPT_PATH=gs://maxtext-model-checkpoints/llama3.1_8b_instruct/2025-10-16/scanned/0/items
   python3 -m maxtext.trainers.post_train.sft.train_sft run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY model_name=$MODEL load_parameters_path=$MAXTEXT_CKPT_PATH per_device_batch_size=1 steps=5
   ```
2. **GRPO on Llama3.1-8b-Instruct**:
   ```bash
   export RUN_NAME=grpo-llama-single-$(date +%Y%m%d%H%M%S)
   export MODEL=llama3.1-8b-Instruct
   export MAXTEXT_CKPT_PATH=gs://maxtext-model-checkpoints/llama3.1_8b_instruct/2025-10-16/scanned/0/items
   python3 -m maxtext.trainers.post_train.rl.train_rl model_name=$MODEL load_parameters_path=$MAXTEXT_CKPT_PATH run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY chips_per_vm=4 rollout_data_parallelism=1 rollout_tensor_parallelism=4 rollout_expert_parallelism=1
   ```
3. **GRPO on Qwen3-0.6B**:
   ```bash
   export RUN_NAME=grpo-qwen-single-$(date +%Y%m%d%H%M%S)
   NEW_MODEL_DESIGN=1 TPU_BACKEND_TYPE=jax python3 -m maxtext.trainers.post_train.rl.train_rl model_name=qwen3-0.6b run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY hbm_utilization_vllm=0.4 rollout_data_parallelism=2 rollout_tensor_parallelism=2 allow_split_physical_axes=true load_parameters_path=gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' vllm_additional_config='{"maxtext_config": {"model_name": "qwen3-0.6b", "log_config": "false"}}'
   ```
4. **vLLM Inference Decoding**:
   ```bash
   python3 -m maxtext.inference.vllm_decode model_name=qwen3-8b tokenizer_path=Qwen/Qwen3-8B load_parameters_path=gs://maxtext-model-checkpoints/qwen3-8b/unscanned/0/items vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' ici_tensor_parallelism=2 ici_data_parallelism=2 hbm_utilization_vllm=0.5 prompt="Suggest some famous landmarks in London." decode_sampling_temperature=0.0 decode_sampling_nucleus_p=1.0 decode_sampling_top_k=0.0 use_chat_template=True scan_layers=False
   ```

## Phase 3: Multi-Host TPU Validation

Execute locally on the runner machine configured with Docker and GKE access. Set up a dedicated environment for the runner.

### 1. Runner Environment Setup
```bash
export VERSION=<Target Release Version>
uv venv --python 3.12 --seed maxtext_runner_$VERSION
source maxtext_runner_$VERSION/bin/activate

gcloud storage cp gs://maxtext_wheels/test/$VERSION/maxtext-$VERSION-py3-none-any.whl .
uv pip install maxtext-$VERSION-py3-none-any.whl[runner] --resolution=lowest
```

### 2. Multi-Host Pre-Training
```bash
# Build and upload image
build_maxtext_docker_image
upload_maxtext_docker_image CLOUD_IMAGE_NAME=maxtext-$VERSION-pre-training
export DOCKER_IMAGE=<Image path output from upload command>

export RUN_NAME=pre-multi-$(date +%Y%m%d%H%M%S)
# Launch XPK workload
xpk workload create --cluster $CLUSTER_NAME --workload pre-train-$RUN_NAME --docker-image $DOCKER_IMAGE --tpu-type $TPU_TYPE --num-slices 1 --command "python3 -m maxtext.trainers.pre_train.train run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY dataset_type=synthetic steps=5 model_name=llama2-7b load_parameters_path=gs://maxtext-model-checkpoints/llama2-7b/2025-01-23-19-26/scanned/0/items" --zone=$ZONE --project=$PROJECT
```
- Monitor the XPK workload status until completion. Verify successful exit logs.

### 3. Multi-Host Post-Training Verification
Execute these distributed verification workloads **strictly sequentially**. Before launching each workload, verify that the cluster queue is clear (`xpk workload list`). Do not launch the next workload until the current workload reaches `COMPLETED` status.

```bash
# Build and upload post-training image
build_maxtext_docker_image WORKFLOW=post-training
upload_maxtext_docker_image CLOUD_IMAGE_NAME=maxtext-$VERSION-post-training
export DOCKER_IMAGE=<Image path output from upload command>

export MODEL=llama3.1-8b-Instruct
export MAXTEXT_CKPT_PATH=gs://maxtext-model-checkpoints/llama3.1_8b_instruct/2025-11-13/pathways/scanned/0/items
export SFT_CKPT_PATH=gs://maxtext-model-checkpoints/llama3.1_8b_instruct/2025-10-16/scanned/0/items
```

#### Test A: SFT McJAX
```bash
# 1. Confirm cluster queue is clear
xpk workload list --cluster $CLUSTER_NAME --project=$PROJECT --zone=$ZONE

# 2. Submit workload
export RUN_NAME=sft-mcjax-$(date +%Y%m%d%H%M%S)
xpk workload create --cluster $CLUSTER_NAME --workload $RUN_NAME --docker-image $DOCKER_IMAGE --tpu-type $TPU_TYPE --num-slices 1 --command "python3 -m maxtext.trainers.post_train.sft.train_sft run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY model_name=$MODEL load_parameters_path=$SFT_CKPT_PATH per_device_batch_size=1 steps=5 hf_access_token=$HF_TOKEN" --zone=$ZONE --project=$PROJECT --priority=high
```
- Monitor `$RUN_NAME` to successful completion before proceeding.

#### Test B: SFT Pathways
```bash
# 1. Confirm previous workload completed and queue is clear
xpk workload list --cluster $CLUSTER_NAME --project=$PROJECT --zone=$ZONE

# 2. Submit workload
export RUN_NAME=sft-pathways-$(date +%Y%m%d%H%M%S)
xpk workload create-pathways --cluster $CLUSTER_NAME --workload $RUN_NAME --docker-image $DOCKER_IMAGE --tpu-type $TPU_TYPE --num-slices 1 --command "JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' python3 -m maxtext.trainers.post_train.sft.train_sft run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY model_name=$MODEL load_parameters_path=$MAXTEXT_CKPT_PATH per_device_batch_size=1 steps=5 hf_access_token=$HF_TOKEN enable_single_controller=True checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False" --zone=$ZONE --project=$PROJECT --priority=high
```
- Monitor `$RUN_NAME` to successful completion before proceeding.

#### Test C: GRPO Pathways
```bash
# 1. Confirm previous workload completed and queue is clear
xpk workload list --cluster $CLUSTER_NAME --project=$PROJECT --zone=$ZONE

# 2. Submit workload
export RUN_NAME=grpo-pathways-$(date +%Y%m%d%H%M%S)
xpk workload create-pathways --cluster $CLUSTER_NAME --workload $RUN_NAME --docker-image $DOCKER_IMAGE --tpu-type $TPU_TYPE --num-slices 1 --command "HF_TOKEN=$HF_TOKEN TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' SKIP_JAX_PRECOMPILE='1' python3 -m maxtext.trainers.post_train.rl.train_rl model_name=$MODEL load_parameters_path=$MAXTEXT_CKPT_PATH run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY chips_per_vm=8 hf_access_token=$HF_TOKEN rollout_data_parallelism=16 rollout_tensor_parallelism=8 rollout_expert_parallelism=1" --zone=$ZONE --project=$PROJECT --priority=high
```
- Monitor `$RUN_NAME` to successful completion before proceeding.

## Phase 4: Benchmark Runner Verification

Validate the repository source builds correctly for benchmarking. Execute locally on runner machine:

```bash
# Clean checkout verification
git clone https://github.com/AI-Hypercomputer/maxtext.git maxtext_bench_test
cd maxtext_bench_test
export VERSION=<Target Release Version>

uv venv --python 3.12 --seed ~/maxtext_bench_runner_$VERSION
source ~/maxtext_bench_runner_$VERSION/bin/activate
uv pip install -e .[runner] --resolution=lowest

# Test Stable Docker Build & Run
build_maxtext_docker_image MODE=stable
python3 -m benchmarks.benchmark_runner xpk --project=$PROJECT --zone=$ZONE --device_type=$TPU_TYPE --num_slices=1 --cluster_name=$CLUSTER_NAME --base_output_directory=$BASE_OUTPUT_DIRECTORY --model_name="llama3_1_70b_8192" --base_docker_image=maxtext_base_image

# Test Nightly Docker Build & Run
build_maxtext_docker_image MODE=nightly
python3 -m benchmarks.benchmark_runner xpk --project=$PROJECT --zone=$ZONE --device_type=$TPU_TYPE --num_slices=1 --cluster_name=$CLUSTER_NAME --base_output_directory=$BASE_OUTPUT_DIRECTORY --model_name="llama3_1_70b_8192" --base_docker_image=maxtext_base_image
```

## Phase 5: Analytical Review & Final Report Compilation

**CRITICAL PLANNER GATE**: Transition your internal execution from active command running to a systematic forensic audit. Before generating the final report, review the exact execution logs, exit codes, and completion statuses of Phases 0 through 4 against the Critical Execution Rules.

Compile a structured summary artifact (`release_report.md`) for the maintainer adhering strictly to the following format:

### 1. Executive Summary & Recommendation
- Declare an explicit recommendation (**NO-GO** by default).
- **Uncompromising Mathematical Gate**: You must **only** recommend **GO** if every single test across all historical phases in the matrix executed to 100% completion and passed perfectly (**PASS**). If any test in the matrix is marked as **SKIPPED**, **FAILED**, **ERROR**, or **TIMED OUT** (even if you believe it was an infrastructure glitch), the recommendation must unconditionally remain **NO-GO**. Do not rationalize or excuse skipped phases.

### 2. Environment Details
- Verified version string, TPU targets, GKE clusters used.

### 3. Phase Matrix & Preserved Logs
- A comprehensive table listing every test run, exact command executed, target hardware, success status, and clickable file hyperlinks directly referencing the preserved output files containing full verbatim execution traces.

### 4. Anomalies & Regressions
- Explicit callouts for any non-zero exits, unexpected warnings, or stack traces. Include verbatim stderr snippets for failures.

### 5. Next Steps
- **If Successful**: Outline instructions for final PyPI publication, Git release tagging, and confirming that the versioned documentation successfully builds and activates on [ReadTheDocs](https://maxtext.readthedocs.io/).
- **If Failed**: Provide a summary of required code fixes or infrastructure adjustments before re-triggering this skill.
