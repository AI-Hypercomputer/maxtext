# Test Plan: Tunix-DPO Implementation Verification

This document outlines the steps to verify that the new Tunix-based DPO implementation in MaxText is correctly integrated and capable of improving model alignment.

## 0. Working Rules & Constraints
*   **Disk Space**: Constantly monitor disk usage (`df -h .`). Stop immediately if available space drops below 10GB.
*   **Environment**: Always use the local virtual environment (`source maxtext_venv/bin/activate`).
*   **Dependencies**: Use `uv pip install` for any new package requirements.
*   **Commits**: Do **not** squash or amend commits unless explicitly requested. Preserve history for audit.
*   **Reproducibility**: Never run ad-hoc commands for major tasks. Always use/update scripts in the `quals/` directory.
*   **Logging**: Redirect high-volume output to `quals/logs/` and checkpoints to GCS to save local disk space.

## 1. Objectives
*   **Correctness**: Verify the DPO training loop runs without errors on TPU/GPU.
*   **Effectiveness**: Demonstrate that `rewards/accuracy` increases and `rewards/margin` becomes positive.
*   **Baseline Comparison**: Compare results against a **Pure-Tunix** reference run.
*   **Integrity**: Ensure the model maintains general reasoning capabilities.

## 2. Experimental Setup
*   **Model**: `Qwen2.5-1.5B-Instruct`
*   **Dataset**: `argilla/distilabel-intel-orca-dpo-pairs`
*   **Hardware**: Cloud TPU (v4-8)
*   **Storage**: `gs://igorts_europe/ttl=30d/dpo_quals/`

## 3. Phase 0: Setup & Reference Benchmark
### A. Checkpoint Conversion
Convert the HuggingFace model to MaxText/Orbax format.
*   **Command**: `bash quals/phase_00_checkpoint_conversion.sh`
*   **Output**: `gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_mcjax_v2/0/items`

### B. Pure-Tunix Reference
Establish a "gold standard" for rewards.
*   **Command**: `python3 quals/phase_00_pure_tunix_ref.py`

## 4. Phase 1: MaxText Baseline Evaluation (Pre-DPO)
Establish baseline performance of the SFT model within MaxText.
*   **Command**: `bash quals/phase_01_maxtext_baseline_eval.sh`
*   **Verification**: Check logs for initial reward accuracy (~37.5% expected).

## 5. Phase 2: DPO Fine-Tuning
Run the main alignment task.
*   **Command**: `bash quals/phase_02_maxtext_dpo_fine_tuning.sh`
*   **Hyperparameters**: 100 steps, LR 1e-6, Beta 0.1, `scan_layers=False`.

## 5.5 Phase 2.5: Prepare Inference Checkpoint
Convert the NNX DPO training checkpoint (with `base.` prefixes) into a clean, replicated Linen checkpoint for MaxEngine.
*   **Command**: `bash quals/phase_02_prepare_inference_ckpt.sh`
*   **Output**: `gs://igorts_europe/ttl=30d/dpo_quals/maxtext_run/dpo-verification-qwen-v3/inference_ckpt/0/items`

## 6. Phase 3: Verification & Success Criteria
### A. Intrinsic Verification
*   **Reward Accuracy**: Should increase to >75%.
*   **Metrics**: Monitor `rewards/accuracy`, `rewards/margin` in TensorBoard.

### B. Extrinsic Verification (Downstream)
1.  **MMLU Stability**:
    *   **Command**: `bash quals/phase_03_mmlu_eval.sh`
2.  **Qualitative Comparison**:
    *   **Command**: `bash quals/phase_03_qualitative_comparison.sh`
    *   **Output**: Compare `quals/logs/sft_responses.log` vs `quals/logs/dpo_responses.log`.

## 7. Bootstrap Guide (In case of crash)
If the VM or session crashes, follow these steps to resume:
1.  **Disk Cleanup**: `rm -rf /home/igorts_google_com/.cache/huggingface/hub/*` (if needed) and check `df -h .`.
2.  **Env Setup**: `source maxtext_venv/bin/activate`.
3.  **Context Recovery**: Read `quals/session_context.md` and `quals/dpo_qual_results.md` to identify the last successful phase.
4.  **Branch Sync**: Ensure you are on `igorts/dpo-feature-integration` and rebase on `origin/main` if stale.
5.  **Resume**: Re-run the script for the failed phase.
