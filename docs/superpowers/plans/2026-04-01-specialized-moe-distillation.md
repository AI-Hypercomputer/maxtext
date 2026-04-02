# Specialized MoE Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a specialized distillation for DeepSeek V2 16B that freezes MLA layers and reconfigures MoE experts (halving routed, doubling shared).

**Architecture:** We use a configuration-driven approach where the Student model is initialized with a different MoE shape. A custom "parameter surgery" step maps Teacher weights to the Student, and the optimizer is masked to only train the MoE layers.

**Tech Stack:** JAX, Flax NNX, MaxText, Optax.

---

### Task 1: Update Distillation Configuration

**Files:**
- Modify: `src/maxtext/configs/post_train/distillation.yml`

- [x] **Step 1: Update the YAML configuration**
Update the Student and Teacher overrides to reflect the new architecture and distillation settings.

```yaml
# src/maxtext/configs/post_train/distillation.yml overrides
student_overrides:
  model_name: "deepseek2-16b"
  num_experts: 16
  shared_experts: 2
  trainable_parameters_mask: ['.*moe_layers.*']
  # Ensure other parameters match 16B Lite
  base_emb_dim: 2048
  base_num_query_heads: 16
  base_num_kv_heads: 16
  base_mlp_dim: 16384
  base_moe_mlp_dim: 4864
  base_num_decoder_layers: 28
  first_num_dense_layers: 14

teacher_overrides:
  model_name: "deepseek2-16b"
  num_experts: 32
  shared_experts: 1
  load_parameters_path: "gs://yujiedeng-maxtext-dev/distillation/converted-from-hf-ds-v2-16b-fixed/0/items"
```

- [x] **Step 2: Commit changes**
```bash
git add src/maxtext/configs/post_train/distillation.yml
git commit -m "config: update distillation specs for specialized MoE"
```

---

### Task 2: Implement Parameter Surgery Logic

**Files:**
- Modify: `src/maxtext/trainers/post_train/distillation/train_distill.py`

- [x] **Step 1: Define `initialize_student_from_teacher` function**
This function will handle the weight copy and MoE expert mapping.

```python
def initialize_student_from_teacher(student_model, teacher_model, config):
    """Performs parameter surgery to initialize student from teacher."""
    max_logging.log("Starting Parameter Surgery: Teacher -> Student...")
    
    # 1. Get states
    s_state = nnx.state(student_model)
    t_state = nnx.state(teacher_model)
    
    # 2. Direct copy for common layers (MLA, Norms, Dense MLP)
    # We use nnx.update to copy weights where names match and shapes are identical
    # For MoE layers, we need manual slicing
    
    def surgery_fn(path, s_param):
        # path is a tuple of keys
        str_path = ".".join(map(str, path))
        
        # If it's a MoE layer, we might need surgery
        if "moe_layers" in str_path:
            # Handle Routed Experts (MoeBlock_0)
            if "MoeBlock_0" in str_path and any(k in str_path for k in ["wi_0", "wi_1", "wo"]):
                t_param = t_state[path]
                # Student 16 experts <- Teacher first 16 experts
                # Shape: [num_experts, emb_dim, intermediate_dim] or sharded equivalent
                return s_param.replace(value=t_param.value[:16, ...])
            
            # Handle Shared Experts
            if "shared_experts" in str_path:
                # Handle specific expert kernels in shared_experts
                if any(k in str_path for k in ["wi_0", "wi_1", "wo"]):
                    # Shared experts in DeepSeek are usually MlpBlock.
                    # If Student shared_experts=2 and Teacher shared_experts=1
                    # Student.shared_experts.wi_0 shape: [2, emb_dim, hidden_dim]
                    # Teacher.shared_experts.wi_0 shape: [1, emb_dim, hidden_dim]
                    # Teacher.MoeBlock_0.wi_0 shape: [32, emb_dim, hidden_dim]
                    
                    t_shared_param = t_state[path]
                    
                    # Find matching routed expert path in teacher
                    # teacher.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.wi_0
                    # student.moe_layers.DeepSeekMoeBlock_0.shared_experts.wi_0
                    
                    # Construct path for routed expert 16
                    routed_path = list(path)
                    # Replace 'shared_experts' with 'MoeBlock_0'
                    for i, p in enumerate(routed_path):
                        if p == 'shared_experts':
                            routed_path[i] = 'MoeBlock_0'
                            break
                    
                    t_routed_param = t_state[tuple(routed_path)]
                    
                    # Combine: [Teacher Shared 0, Teacher Routed 16]
                    combined_val = jnp.concatenate([
                        t_shared_param.value,
                        t_routed_param.value[16:17, ...]
                    ], axis=0)
                    
                    return s_param.replace(value=combined_val)
        
        # Default: Try to copy from teacher if path exists and shape matches
        if path in t_state:
            t_param = t_state[path]
            if s_param.value.shape == t_param.value.shape:
                return s_param.replace(value=t_param.value)
        
        return s_param

    new_s_state = jax.tree_util.tree_map_with_path(surgery_fn, s_state)
    nnx.update(student_model, new_s_state)
    max_logging.log("Parameter Surgery Complete.")
```

- [x] **Step 2: Call the surgery function in `train_distill`**
Insert the call after model initialization but before training starts.

- [x] **Step 3: Commit changes**
```bash
git add src/maxtext/trainers/post_train/distillation/train_distill.py
git commit -m "feat: implement parameter surgery for specialized MoE"
```

---

### Task 3: Implement Selective Freezing in Trainer

**Files:**
- Modify: `src/maxtext/trainers/post_train/distillation/train_distill.py`

- [x] **Step 1: Modify `MaxTextDistillationTrainer.__init__`**
Apply the `trainable_parameters_mask` to the optimizer initialization.

```python
# Inside MaxTextDistillationTrainer.__init__
import re

mask = training_config.get("trainable_parameters_mask", [])
if mask:
    def is_trainable(path, _):
        str_path = ".".join(map(str, path))
        return any(re.match(p, str_path) for p in mask)
    
    wrt = nnx.filter(student_model, is_trainable)
else:
    wrt = nnx.Param # Default
    
self.optimizer = nnx.Optimizer(student_model, optimizer, wrt=wrt)
```

- [x] **Step 2: Commit changes**
```bash
git add src/maxtext/trainers/post_train/distillation/train_distill.py
git commit -m "feat: add selective freezing to distillation trainer"
```

---

### Task 4: Verification and Dry Run

**Files:**
- Create: `tests/distillation/test_specialized_moe.py`

- [x] **Step 1: Write a unit test for parameter surgery**
Verify that the slicing logic works as expected.

- [x] **Step 2: Run the test**
Run: `pytest tests/distillation/test_specialized_moe.py`

- [x] **Step 3: Commit test**
```bash
git add tests/distillation/test_specialized_moe.py
git commit -m "test: add verification for MoE surgery"
```
