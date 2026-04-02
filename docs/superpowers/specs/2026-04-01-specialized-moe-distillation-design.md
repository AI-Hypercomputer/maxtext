# Design Spec: Specialized MoE Distillation for DeepSeek V2 (16B)

This document specifies the architectural and procedural changes required to implement a specialized distillation of the DeepSeek V2 model. The goal is to stabilize distillation by freezing the Multi-Head Latent Attention (MLA) layers and reconfiguring the Mixture-of-Experts (MoE) capacity.

## 1. Architectural Goals
The Student model will be a modified version of the DeepSeek V2 16B (Lite) architecture, specialized for higher shared-expert capacity and lower routed-expert count.

### 1.1 Teacher Model (Frozen)
*   **Checkpoint:** `gs://yujiedeng-maxtext-dev/distillation/converted-from-hf-ds-v2-16b-fixed/0/items`
*   **Architecture:** DeepSeek V2 16B (Lite)
    *   **Routed Experts:** 32
    *   **Shared Experts:** 1
    *   **MLA:** Full (16 Query/KV Heads, 512/128 ranks)
    *   **Layers:** 28 (14 Dense + 14 MoE)

### 1.2 Student Model (Trainable)
*   **Architecture:** Specialized DeepSeek V2 16B
    *   **Routed Experts:** 16 (Halved from 32)
    *   **Shared Experts:** 2 (Doubled from 1)
    *   **MLA:** Full (Identical to Teacher)
    *   **Layers:** 28 (Identical to Teacher)

## 2. Parameter Surgery (Initialization)
Since the Student and Teacher have incompatible MoE shapes, we cannot use standard checkpoint restoration. We will implement a custom initialization phase (`initialize_student_from_teacher`) in `src/maxtext/trainers/post_train/distillation/train_distill.py`.

### 2.1 Weight Mapping Logic
After both models are initialized on the device mesh:
1.  **MLA & Norms:** Perform a 1-to-1 weight copy for all `self_attention`, `layer_norm`, `token_embedder`, and `dense_layers.mlp` parameters.
2.  **MoE Routed Experts:**
    *   `student.moe_layers.MoeBlock_0[0:16]` = `teacher.moe_layers.MoeBlock_0[0:16]`
3.  **MoE Shared Experts:**
    *   `student.moe_layers.shared_experts[0]` = `teacher.moe_layers.shared_experts[0]`
    *   `student.moe_layers.shared_experts[1]` = `teacher.moe_layers.MoeBlock_0[16]` (Using the 17th routed expert from the teacher as the second shared expert).

## 3. Training & Freezing Logic

### 3.1 Selective Optimization
We will enforce freezing by masking the parameters provided to the optimizer.
*   **Optimizer:** `nnx.Optimizer` will be initialized with a `wrt` (write) filter.
*   **Frozen:** `self_attention`, `layer_norm`, `token_embedder`, `dense_layers.mlp`.
*   **Trainable:** `moe_layers` (specifically the routed and shared expert kernels/biases).

### 3.2 Configuration
The main entry point will be `src/maxtext/configs/post_train/distillation.yml`. It will be updated to:
*   Set `student_overrides` with `num_experts: 16` and `shared_experts: 2`.
*   Set `teacher_overrides` with `num_experts: 32` and `shared_experts: 1`.
*   Include `trainable_parameters_mask: ['.*moe_layers.*']`.

## 4. Verification Plan
1.  **Initialization Check:** Verify that Student parameters after surgery match the expected slices of Teacher parameters.
2.  **Freezing Check:** Verify that gradients for MLA and Norm parameters are `None` or zero during training.
3.  **Loss Check:** Monitor `distill/total_loss` and ensure it starts at a reasonable baseline (not zero, but not diverging).
