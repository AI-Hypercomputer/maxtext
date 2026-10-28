# Team Sync, PR Cherry-Picks (`#5706`, `#5828`, `#5705`), and `>200B` Multi-Slice Experiment Plan

## 1. Summary of Team Alignment & Experiment Goals

### 1.1 Action Items from Team Discussion (Lance Wang, Hsuan-Lun Chiang, Mesakh Christian)

| Item | Requirement | Current Status | Branch / Location |
| :---: | :--- | :---: | :--- |
| **1** | **Create a branch and cherry-pick Lance's 3 `safetensors_dynamic` PRs**:<br>• **#5706**: Fix `safetensors_dynamic` loading for `gpt-oss`, `Qwen3.5`, `Qwen3-Next`, and `Gemma` (`df2eacf0d`)<br>• **#5828**: Fix `safetensors_dynamic` load timeout & inner MoE stack HBM on multi-hosts (`1f06b304d`, `335d5e880`)<br>• **#5705**: Support `safetensors_dynamic` in `from_pretrained` and with `enable_checkpointing=False` (`1a89e9dfc`) | **Completed & Pushed** | `origin/docs/checkpoint-load-benchmark` |
| **2** | **Create portable wrapped script(s) & resources** so the team can run the exact same tests across different machines, clusters, and environments, and upload them to the branch from (1). | **Completed & Pushed** | [`checkpoint_load_benchmark/`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark) on `origin/docs/checkpoint-load-benchmark` |
| **3** | **Verify Before vs. After on Large (`>200B`) & Multi-Slice Models**:<br>• **Without fixes (`origin/main`)**: Models fail (OOM, multi-host 30-min Orbax load timeout due to strided inner-axis byte over-reads, or crash on composite keys / multi-host NumPy hooks).<br>• **With fixes (`#5706 + #5828 + #5705`)**: Models load and convert successfully via `safetensors_dynamic`. | **In Progress / Ready for Multi-Host `>200B` Run** | Hand-off summary below |

> [!IMPORTANT]
> **Guideline for Additional Fixes to `#5706`, `#5828`, `#5705`**:
> If further code fixes are needed in Lance's PRs during testing, push additional commits directly to the original PR branches (**do not squash**) and ensure all pre-commit linters (`pyink`, `pylint`, `codespell`) and GitHub unit tests pass, as those PRs are currently under review by Abhinav.

---

## 2. What Each Cherry-Picked PR Fixes in `safetensors_dynamic`

1. **PR #5706 — `fix(load_dynamic): Composite keys, multi-host hooks, legacy keys`**:
   - **Composite Keys**: Supports both tuple HF source keys (multiple HF tensors fused into one leaf, e.g., `qwen3.5` `in_proj_qkvz`) and tuple MaxText keys (one HF tensor split into multiple leaves, e.g., `gpt-oss` and `qwen3.5` `gate_up_proj`, `qwen3-vl` vision `qkv`).
   - **Multi-Host NumPy Hooks**: When a NumPy parameter hook fails on a non-addressable multi-host `jax.Array` (e.g., `gemma2` / `gemma4` / `qwen3` `pad_hf_embedding_layer`), gathers the tensor in `<=1 GiB` slices along axis 0 (`_gather_in_slices`) and places the result shard-by-shard via `jax.make_array_from_callback` (avoiding `jax.device_put`'s full-device all-gather HBM spike).
   - **Model Mappings & Collections**: Fixes `qwen3-480b-a35b` mapping key name, adds missing `HF_IDS` entries (`qwen3-1.7b`, `qwen3-*-base`), pads short `PartitionSpec`s, and handles `MoEBiasVar` / `Tid2EidVar` collections for DeepSeek models.

2. **PR #5828 — `Fix safetensors_dynamic load timeout on multi hosts` (2 commits)**:
   - **Commit 1 (`Shard safetensors_dynamic reads along the leading axis on multi-host`)**:
     - Replaces Orbax's `construct_maximal_shardings` (which splits 2D weights along inner `axis=1`, creating thousands of strided byte reads and up to `60x` file over-read across hosts) with `_leading_axis_shardings` so every host reads **1 contiguous byte range per tensor**.
     - Sets `max_over_read_ratio=1.0`, disables `gcsfs` 5 MiB read-ahead during `.safetensors` reads, restricts HF Hub file listing to host 0 (avoiding HTTP 429 on 128 hosts), and slice-reshards arrays `>1 GiB` (`_reshard`).
   - **Commit 2 (`Keep inner MoE stacks in the read layout until the final reshard`)**:
     - Prevents inner MoE stacks from being prematurely placed onto partial `sharding_at(level)` specs that replicate across the `expert` axis (`16x` HBM blowup when `ici_expert_parallelism=16`).

3. **PR #5705 — `fix: Support safetensors_dynamic outside the training restore`**:
   - Initializes `jax.distributed` on multi-host TPU whenever a run loads checkpoints (`load_parameters_path`, `load_full_state_path`, or `safetensors_dynamic`), even when `enable_checkpointing=False`.
   - Routes `from_pretrained` through `checkpointing.load_state_if_possible` when `source_checkpoint_layout=safetensors_dynamic`.

---

## 3. Can `bodaborg-v5p-nap` Run `>200B` Models?

**Yes — `bodaborg-v5p-nap` (`gke_cloud-tpu-shared-capacity_europe-west4_bodaborg-v5p-nap`) CAN run `>200B` models, provided we request a multi-host TPU slice (`>= 4` hosts / `v5p-32` or `v5p-128`) instead of a 1-node or 2-node `v5p-8` (`2x2x1`) slice.**

### 3.1 Why 1–2 `v5p-8` (`2x2x1`) Nodes OOM on `>200B` Models
Each individual TPU v5p VM node (`ct5p-hightp-4t`, 4 TPU v5p chips) has:
- **Host RAM**: `270.25 GB` (`~245–252 GB` usable before Kubernetes container cgroup `OOMKilled / Exit Code 137`)
- **TPU HBM**: `380 GB` (`4 chips × 95 GB`)

Therefore:
- On **1 node (`v5p-8`, `2x2x1`)**, total Host RAM is `270 GB` and HBM is `380 GB` $\rightarrow$ max BF16 model size is **`~100B` (`~200 GB` BF16)** (`gpt-oss-120b` at `233.66 GB` OOMs on 1 node).
- On **2 nodes (`2 × v5p-8` or `v5p-16`, `2x2x2`)**, total Host RAM is `540 GB` and HBM is `760 GB` $\rightarrow$ `gpt-oss-120b` (`233.66 GB` BF16) **succeeds** (`133.9–206.1 GB` RAM/node), and `qwen3-235b-a22b` (`~470 GB` BF16) is right at the 2-node ceiling (`~235 GB`/node), while `>=284B` models (`582 GB` to `3,198 GB` BF16) OOM on 2 nodes.

### 3.2 Available Multi-Host Topologies on `bodaborg-v5p-nap`
`bodaborg-v5p-nap` has **Node Auto-Provisioning (NAP)** enabled (`nap-ct5p-hightp-4t-*`) and actively runs large multi-host v5p slices:

| Topology Label (`cloud.google.com/gke-tpu-topology`) | Accelerator Type | Hosts (VMs) per Slice | TPU Chips (`google.com/tpu`) | Aggregate Host RAM / Slice | Aggregate TPU HBM / Slice | Max BF16 Model Capacity (`safetensors_dynamic`) | Active Pools on `bodaborg-v5p-nap` |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`2x2x1`** | `v5p-8` | **1 host** | `4` chips | `270.25 GB` | `380 GB` | Up to **`~70B–100B`** (`llama3.1-70b`) | `nap-ct5p-hightp-4t-1n047s7c` (2 nodes) |
| **`2x2x2`** | `v5p-16` | **2 hosts** | `8` chips | `540.50 GB` | `760 GB` | Up to **`~120B–200B`** (`gpt-oss-120b`) | `nap-ct5p-hightp-4t-1kcjfu9q` (2 nodes) |
| **`2x2x4`** | `v5p-32` | **4 hosts** | `16` chips | **`1,081.00 GB` (`1.08 TB`)** | **`1,520 GB` (`1.52 TB`)** | Up to **`~300B`** (`qwen3-235b-a22b`, `deepseek4-284b`) | Auto-provisioned via NAP |
| **`4x4x4`** | `v5p-128` | **16 hosts** | `64` chips | **`4,324.00 GB` (`4.32 TB`)** | **`6,080 GB` (`6.08 TB`)** | Up to **`~1,000B`** (`qwen3-480b-a35b`, `deepseek3-671b`, `deepseek3.2-671b`) | `nap-ct5p-hightp-4t-101hfhus`, `-dsfl2a0b`, `-obg60z1c` (3 slices = 48 nodes) |
| **`4x4x8`** | `v5p-256` | **32 hosts** | `128` chips | **`8,648.00 GB` (`8.65 TB`)** | **`12,160 GB` (`12.16 TB`)** | Up to **`~2,000B`** (`deepseek4-1600b`) | `nap-ct5p-hightp-4t-6p4an9cr` (1 slice = 32 nodes) |

> [!TIP]
> **Why `safetensors_dynamic` Succeeds on Multi-Host Slices Where `to_maxtext` Fails**:
> - **`to_maxtext` Step 2** is a single-process offline CPU script (`to_maxtext.py`); even on a `4x4x4` (`v5p-128`) slice, it can only use **1 host's `270 GB` RAM**, so it permanently OOMs on any model `>=120B` (`gpt-oss-120b`, `qwen3-235b-a22b`, `deepseek3-671b`, etc.).
> - **`safetensors_dynamic`** (with Lance's `#5706 + #5828 + #5705`) shards the `.safetensors` download and transformation across **all hosts in the TPU slice / multi-slice mesh**, dividing per-host RAM and HBM by the number of hosts (`16` hosts on `v5p-128`, `32` hosts on `v5p-256`).

---

## 4. Progress Summary & Hand-Off for Multi-Slice / `>200B` Runs

### 4.1 Completed So Far (`run_20261023_032224`)
1. **Branch & Scripts Ready**:
   - Branch **`origin/docs/checkpoint-load-benchmark`** has `#5706`, `#5828`, `#5705` cherry-picked on top of `origin/main`, plus the complete [`checkpoint_load_benchmark/`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/checkpoint_load_benchmark) suite.
2. **48 Configurations Benchmarked on 1 & 2 `v5p-8` (`2x2x1`) Slices**:
   - **Tier 0 (`<=32B`: `qwen3-0.6b`, `gemma2-2b`, `deepseek2-16b`, `qwen3-30b-a3b`, `qwen3-32b`)**: All 20 configs succeeded (`safetensor` `1.20x–3.05x` faster E2E on single-slice).
   - **Tier 1 (`47B–70B`: `mixtral-8x7b`, `llama3.1-70b`)**: All 8 configs succeeded on `bodaborg-v5p-nap` (`safetensor` `1.10x–1.47x` faster than `to_maxtext` across all 4 `llama3.1-70b` configs).
   - **Tier 2 (`120B` OOM Boundary: `gpt-oss-120b`, `233.66 GB` BF16)**:
     - `to_maxtext` Step 2 (`scanned` & `unscanned`): **OOM (`Exit Code 137`)** at `251.69 GB` RAM.
     - `safetensors_dynamic` (`1 slice`, `v5p-8`): **OOM (`Exit Code 137`)** at `253.06 GB` RAM.
     - `safetensors_dynamic` (`2 slices`, `2 × v5p-8`): **SUCCEEDED** in both `scanned` (`3,061.14s`, `Peak RAM: 206.09 GB/slice`) and `unscanned` (`3,004.55s`, `Peak RAM: 133.86 GB/slice`).
   - **Tier 3 (`284B–1600B` DeepSeek OOM Boundary on 1 & 2 `v5p-8` Nodes: `deepseek4-284b`, `deepseek3-671b`, `deepseek3.2-671b`, `deepseek4-1600b`)**:
     - Logged exact OOM points (`Exit Code 137` at `240–246 GB` RAM) on 1 and 2 `v5p-8` nodes, confirming they require `>=4` to `>=16` hosts.

### 4.2 Next Steps for `>200B` Before/After Verification (Without vs. With `#5706 + #5828 + #5705`)
To run the `>200B` Before/After experiment on `bodaborg-v5p-nap` (e.g., `qwen3-235b-a22b` at `470 GB` BF16 or `qwen3-480b-a35b` at `960 GB` BF16):
1. **Use a Multi-Host TPU Slice (`2x2x4` `v5p-32` [4 hosts] or `4x4x4` `v5p-128` [16 hosts])**:
   - **Without fixes (`origin/main`)**: Run `safetensors_dynamic` $\rightarrow$ verify/log the strided-read OOM or 30-min Orbax timeout / hook crash.
   - **With fixes (`origin/docs/checkpoint-load-benchmark`)**: Run `safetensors_dynamic` $\rightarrow$ verify clean multi-host load and record load time & peak host RAM.
