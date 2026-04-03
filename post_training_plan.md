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

### A. The Model Wrapper (The "Naming Bridge")
Tunix trainers expect a generic model interface. We should formalize the `ModelWrapper` we started building.
*   **Action:** Create a standard `MaxTextTunixWrapper` in `src/maxtext/trainers/post_train/utils.py` that handles mapping generic names (like `positions`, `attention_mask`) to model-specific names (like `decoder_input_tokens_positions`).

### B. Sharding-Aware Initialization
Tunix's `PeftTrainer` currently makes assumptions about sharding that clash with MaxText's more advanced SPMD rules (e.g., the `norm` axis issue and scalar optimizer states).
*   **Action:** Contribute to Tunix to make its internal sharding logic optional or configurable.
*   **Action:** Ensure MaxText always provides a "Pre-Sharded" state to Tunix, and Tunix should respect existing sharding rather than attempting to re-apply it.

### C. Standardized Data Schema
We need a unified "Post-Training Data Schema" that both projects understand.
*   **DPO Schema:** `prompt_ids`, `chosen_ids`, `rejected_ids` + corresponding masks.
*   **RL Schema:** `queries`, `responses`, `rewards`.
*   **Action:** Implement `MaxTextDPOTprep` and `MaxTextRLPrep` transforms in the MaxText pipeline that output exactly what Tunix trainers expect.

## 4. Specific Collaboration Opportunities

### Contribution to Tunix
1.  **Refactor `_shard_optimizer`:** Modify Tunix to check if an optimizer is already sharded before attempting to apply `with_sharding_constraint`.
2.  **Generalized Keyword Arguments:** Update Tunix's `get_per_token_logps` to accept a mapping for keyword arguments, avoiding the need for a manual `ModelWrapper` for every new model.

### Enhancement in MaxText
1.  **Alignment-Aware Hooks:** Formalize the `SFTTrainingHooks` into a more general `PostTrainingHooks` system that detects the algorithm (SFT vs DPO vs RL) and adjusts metric calculation accordingly.
2.  **Parameter-Efficient Fine-Tuning (PEFT):** Leverage Tunix's LoRA/PEFT logic while applying MaxText's optimized kernel implementations.

## 5. Roadmap for DPO Integration (Immediate Next Steps)
1.  **Finalize the `ModelWrapper`:** Fix the "too many values to unpack" error by ensuring the wrapper returns only what Tunix needs (logits).
2.  **Formalize the "No-Op" Sharding Override:** Instead of a lambda hack, create a proper `MaxTextDPOTrainer` subclass that overrides `_shard_optimizer` cleanly.
3.  **Unified Config:** Allow users to specify `post_training_flavor: tunix_dpo` in `dpo.yml` to automatically trigger these bridge behaviors.
