# Brainstorming: MaxText & Tunix Post-Training Integration Plan

## 1. Executive Summary
The goal is to provide a best-in-class post-training suite (SFT, DPO, RLHF, GRPO) that scales to the largest models and TPU slices. 

Instead of maintaining duplicate implementations of complex alignment algorithms, we will establish a **"Hybrid Core"** architecture:
*   **MaxText** acts as the **Performance Engine**: Providing optimized model implementations (NNX), robust sharding/SPMD rules, and high-throughput data loading.
*   **Tunix** acts as the **Algorithmic Orchestrator**: Providing the training loops, specialized loss functions (DPO, PPO), and alignment-specific metrics.

## 2. Shared Responsibilities & Strengths

| Feature | MaxText Strength | Tunix Strength | Recommended Primary |
| :--- | :--- | :--- | :--- |
| **Model Arch** | highly optimized, NNX-based, TPU-aware | research-flexible | **MaxText** |
| **Sharding** | Robust logical-to-physical SPMD rules | Basic/Standard sharding | **MaxText** |
| **Dataloading** | Multi-host Grain integration | HF Datasets convenience | **Collaborative** (MaxText Grain + Tunix Prep) |
| **Loss Functions**| Standard Cross-Entropy | DPO, ORPO, PPO, GRPO | **Tunix** |
| **Metrics** | Goodput, Hardware utilization | KL-Divergence, Rewards, Accuracy | **Tunix** (Loop) + **MaxText** (System) |

## 3. The "Bridge" Architecture (Implementation Strategy)

To make these two projects work together without "technical friction," we should standardize the following interfaces:

### A. The Model Adapter (Unified Naming Bridge)
We discovered that `src/maxtext/integration/tunix/tunix_adapter.py` already contains a robust `TunixMaxTextAdapter`.
*   **Current State:** Used effectively in RL (`train_rl.py`).
*   **Action:** Refactor SFT (`train_sft.py`) and DPO (`train_dpo.py`) to use this same adapter instead of ad-hoc wrappers. This ensures that any model supported by MaxText is immediately compatible with all Tunix trainers.

### B. Sharding-Aware Initialization
Tunix's `PeftTrainer` currently makes assumptions about sharding that clash with MaxText's more advanced SPMD rules (e.g., the `norm` axis issue and scalar optimizer states).
*   **Current State:** Handled via manual "pre-sharding" and no-op overrides in DPO.
*   **Action:** Move this logic into a base `MaxTextTunixTrainer` class or a utility function used by all post-training scripts.
*   **Action:** Contribute to Tunix to make its internal `_shard_optimizer` check for existing sharding before applying constraints.

### C. Standardized Data Schema (The "Input Bridge")
MaxText's multi-host Grain loader requires numeric arrays, while Tunix often expects strings.
*   **Current State:** SFT/DPO/RL each handle this differently.
*   **Action:** Standardize on a "Pre-tokenized numeric schema" where MaxText performs tokenization and padding (using DPO-aware left-padding when needed) and provides the `_ids` and `_mask` columns Tunix expects for pre-tokenized input.

## 4. Documentation Strategy

Existing documentation is fragmented (`docs/tutorials/posttraining/sft.md`, `rl.md`, etc.).
*   **Action:** Create a unified `post_training_overview.md` that explains the MaxText-Tunix relationship (MaxText=Engine, Tunix=Brain).
*   **Action:** Ensure all tutorials consistently mention the `maxtext[tpu-post-train]` installation requirement.

## 4. Collaborative Enhancements (Modifications to Tunix)

To further reduce the "glue code" in MaxText, we should upstream the following improvements to the Tunix library:

### A. Flexible Sharding in `PeftTrainer`
Tunix's `_shard_optimizer` currently forces sharding constraints that can crash on pre-sharded MaxText states (especially with scalar values).
*   **Action:** Modify `tunix/sft/peft_trainer.py` to only apply `with_sharding_constraint` if the optimizer is not already sharded or if a specific flag is set.

### B. Generalized Model Call Interface
Tunix's `get_per_token_logps` hardcodes argument names like `positions` and `attention_mask`.
*   **Action:** Update `tunix/rl/common.py` to allow passing a `name_mapping` dictionary. This would allow MaxText to tell Tunix: "Use `decoder_positions` instead of `positions`."

## 5. Cleanup: Deleting Legacy Post-Training Support

As we transition to the Tunix-based "Hybrid Core" architecture, we should remove the legacy, non-Tunix implementations from MaxText to reduce maintenance burden.

### A. Remove Legacy DPO
The existing internal DPO implementation is fragmented and harder to maintain than the Tunix version.
*   **Action:** Delete `src/maxtext/trainers/post_train/dpo/dpo_utils.py`.
*   **Action:** Remove DPO-specific branches and imports in:
    *   `src/maxtext/trainers/pre_train/train.py`
    *   `src/maxtext/utils/train_utils.py`
    *   `src/MaxText/__init__.py`
*   **Action:** Deprecate legacy DPO-specific configuration parameters in `src/maxtext/configs/base.yml` once the Tunix bridge is stable.

## 6. Roadmap for DPO Integration (Immediate Next Steps)
1.  **Finalize the `ModelWrapper`:** Fix the "too many values to unpack" error by ensuring the wrapper returns only what Tunix needs (logits).
2.  **Formalize the "No-Op" Sharding Override:** Instead of a lambda hack, create a proper `MaxTextDPOTrainer` subclass that overrides `_shard_optimizer` cleanly.
3.  **Unified Config:** Allow users to specify `post_training_flavor: tunix_dpo` in `dpo.yml` to automatically trigger these bridge behaviors.
