# DPO Qualification Test Plan: Tunix-MaxText Integration

This document establishes the testing roadmap, execution phases, and strict working constraints for validating the new Tunix-based DPO post-training implementation in MaxText.

---

## 1. Strict Rules of Engagement & Operational Constraints

All execution within this and subsequent AI agent sessions must strictly adhere to these instructions. There are no exceptions.

1. **Strict Reproducibility (No Ad-hoc Commands):**
   * Every execution of a MaxText training, inference, or evaluation run must correspond to a dedicated shell (`.sh`) or Python (`.py`) script under the `quals/` directory.
   * Ad-hoc CLI command execution is strictly prohibited. Any configuration modifications must be updated directly within the script files to maintain an auditable history.

2. **Centralized & Persistent Logging:**
   * All process output, stdout, stderr, and evaluation logs must be directed to files within the local `quals/logs/` directory on the Cloudtop machine.
   * Never execute commands without piping outputs to a corresponding log file.
   * **Ephemeral VM Policy:** Always treat the remote TPU VM as an ephemeral machine. When running workloads on the remote VM via SSH, configure the redirection *externally* on the local Cloudtop shell (e.g. `gcloud compute ... ssh ... --command="..." > quals/logs/... 2>&1`) so that the log stream is saved locally on Cloudtop rather than on the remote VM.

3. **Identify Bugs, Do Not Fix:**
   * The primary objective of qualification testing is to discover bugs, regression errors, or configuration mismatches.
   * **Do not attempt to patch, debug, or fix any codebase problems** unless explicitly instructed by the user.

4. **Fail Loudly & Report Immediately:**
   * If any component fails or behaves unexpectedly (e.g., checkpoint loader issues a warning, keys fail to match, shapes mismatch, or execution crashes), **halt execution immediately and report it to the user.**
   * Do not attempt to bypass errors with silent code fallbacks or temporary debugging workarounds.

5. **No Shortcuts or Simplifications:**
   * Never apply shortcuts to circumvent environmental constraints.
   * For example:
     * If GCS bucket writing is blocked, **do not** redirect outputs to `/tmp/` or local directories.
     * If a run encounters Out-Of-Memory (OOM) or timeout errors, **do not** reduce the sequence length, batch size, or model configuration.
   * Report constraints directly to the user for discussion.

6. **Disk Space Monitoring:**
   * Always monitor disk space constraints (`df -h`) on both the local cloudtop and the remote TPU VM.
   * If available disk space drops below 10GB on either host, pause operations immediately and perform cleanup (e.g., purging old caches or local temporary checkpoint copies) before proceeding.

## 2. Scrutiny & Adversarial Objectives

The primary mandate of this qualification process is **not to achieve a "green light" or prove success at any cost.** Instead, the AI agent must aggressively scrutinize the DPO integration, assuming bugs exist until proven otherwise.

* **Aggressively Audit Checkpoint Integrity:** Hunt for key mismatches, omitted weights, shape anomalies, or silent initializations in the NNX-to-Linen loading pathway. Treat any load warning or fallback weight initializations as a major bug.
* **Adversarially Verify Behavioral Steerability:** Actively test whether the DPO alignment has mathematically and behaviorally altered the model. If the SFT baseline and DPO checkpoints have low parameter divergence or output near-identical decodings, it must be treated as a silent integration failure.
* **Probe for Downstream Regression & Drift:** Scrutinize general knowledge benchmarks (MMLU) to detect subtle degradation, domain-specific collapses, or catastrophic knowledge drift post-alignment.

---

## 3. Core Verification Phases

### Phase 1: Setup and Checkpoint Conversion
Establish the baseline model in MaxText format.
* **Script:** `quals/phase_01_checkpoint_conversion.sh`
* **Checkpoints:** Converted baseline located at `gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_mcjax_v2/0/items`

### Phase 2: SFT Baseline Metrics Evaluation
Establish baseline performance of SFT model (e.g. initial reward accuracy ~37.5% expected) by executing a single-step DPO training run.
* **Script:** `quals/phase_02_maxtext_baseline_eval.sh`

### Phase 3: Logit Parity Cross-Framework Scrutiny (PyTorch/TRL vs. MaxText)
Perform an adversarial mathematical cross-check to prove that MaxText attention layout, positional scaling, and DPO loss computation are perfectly aligned with PyTorch/TRL reference implementations.
* **Method:**
  1. Feed an identical fixed sequence data batch (containing chosen/rejected prompt inputs) to both PyTorch (Hugging Face AutoModelForCausalLM) and MaxText/Tunix baseline models.
  2. Extract pre-update reference logit arrays for SFT reference, chosen, and rejected sequences.
  3. Run a single forward pass and optimization step on both frameworks with the identical batch.
  4. Extract post-update logits, DPO loss, and gradient magnitudes.
  5. Compare values using absolute/relative tolerance checks.
* **Loud-Failure Assertion:** The Mean Absolute Error (MAE) between baseline logits must satisfy $\text{MAE} < 10^{-4}$ (under `bfloat16` precision limits). A failure indicates coordinate/attention scaling bugs, padding mistakes, or sign flips, which must halt qualification.
* **Script:** `quals/phase_03_logit_parity_check.py` (to be created when executing this phase)

### Phase 4: Tunix DPO Post-Training
Run the main preference optimization training loop.
* **Script:** `quals/phase_04_maxtext_dpo_fine_tuning.sh`

### Phase 5: Inference Checkpoint Extraction
Extract and shard the NNX checkpoint parameters for Linen inference.
* **Script:** `quals/phase_05_prepare_inference_ckpt.sh`

### Phase 6: Multi-Layered Automated Validation & Benchmarks
Execute strict verification checks before downstream scoring.

#### A. Mathematical Parameter Divergence
Confirm that the training weights have successfully diverged from the SFT baseline.
* **Script:** `python3 quals/validate_divergence.py --sft_path <SFT_PATH> --dpo_path <DPO_PATH>`
* **Constraint:** Exit code `0` represents successful divergence. Exit code `2` signals failure (checkpoints are mathematically identical).

#### B. MMLU Downstream Benchmarking
Benchmark general knowledge stability.
* **Script:** `quals/phase_06_mmlu_eval.sh`
* **Assertion:** Gated by `validate_divergence.py`.

#### C. Qualitative Behavioral Validation
Evaluate decoding responses and semantic steering.
* **Script:** `quals/phase_06_qualitative_comparison.sh`
* **Assertion:** Automatically executes `quals/compare_quals.py` to compute Jaccard word similarity and assert correct terminology alignment. Failures (exit codes `3` and `4`) indicate zero behavioral steering.
