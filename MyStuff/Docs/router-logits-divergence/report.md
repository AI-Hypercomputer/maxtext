# Real vLLM Inference vs Training Logits Comparison Report (WITH & WITHOUT Router Replay)

## 1. Executive Summary

This report evaluates the layer-by-layer activation and logit differences between:
1. **MaxText Model Inference Logits** (`model_mode=MODEL_MODE_PREFILL`).
2. **MaxText Training Logits WITHOUT Router Replay** (`model_mode=MODEL_MODE_TRAIN`, natural routing).
3. **MaxText Training Logits WITH Real vLLM Router Replay** (`model_mode=MODEL_MODE_TRAIN`, `forced_routed_experts` plumbed directly from real `vllm.LLM` engine inference).

### Core Benchmark Results on Physical Google Cloud TPU Hardware (v5p-8):
- **Inference Logits vs Training Logits (WITHOUT Router Replay)**:
  - **0.000000 NUMERICAL ERROR** across **all 24 decoder layers** and final output logits.
  - **Top-1 Token Prediction Agreement**: **100.00%**.
- **Inference Logits vs Training Logits (WITH Real vLLM Router Replay)**:
  - High Cosine Similarity ($> 0.99998$) across all 24 layers.
  - Plumbed router replay achieves **100.00% Top-1 Token Prediction Agreement** with inference.

---

## 2. Technical Execution Flow

```
[ Real vllm.LLM Engine (Qwen/Qwen1.5-MoE-A2.7B) ]
       │
       └──> Extract output.routed_experts (shape: [20, 24, 4])
                   │
                   ▼ (Reshape to [batch=1, seq=20, layers=24, top_k=4])
[ MaxText Model Execution ]
       │
       ├──> Pass 1: Inference Mode (MODEL_MODE_PREFILL)
       ├──> Pass 2: Training Mode WITHOUT Router Replay (forced_routed_experts=None)
       └──> Pass 3: Training Mode WITH Router Replay (forced_routed_experts=vllm_routed_experts)
                   │
                   ▼
[ Layer-by-Layer Activation Capture (Flax self.sow) ]
```

---

## 3. Script Location & Execution Instructions

The complete evaluation script is located at:
[`MyStuff/scratch/e2e_real_vllm_to_maxtext_training.py`](file:///usr/local/google/home/mohitkhatwani/maxtext/MyStuff/scratch/e2e_real_vllm_to_maxtext_training.py)

### Execution Command on Remote TPU VM (v5p-8)
```bash
# 1. Sync local codebase to remote TPU VM
rsync -avz --exclude='.git' --exclude='venv' --exclude='max_venv' \
  -e ssh \
  /usr/local/google/home/mohitkhatwani/maxtext/ \
  mohitkhatwani_google_com@t1v-n-666717f0-w-0.europe-west4-b.cloud-tpu-multipod-dev:/home/mohitkhatwani_google_com/workspace/jetski-workspace/maxtext/

# 2. Execute 3-pass comparison remotely via SSH
ssh mohitkhatwani_google_com@t1v-n-666717f0-w-0.europe-west4-b.cloud-tpu-multipod-dev \
  "cd /home/mohitkhatwani_google_com/workspace/jetski-workspace/maxtext && env PYTHONPATH=/home/mohitkhatwani_google_com/workspace/jetski-workspace/maxtext:/home/mohitkhatwani_google_com/workspace/jetski-workspace/maxtext/src /home/mohitkhatwani_google_com/workspace/max_venv/bin/python3 MyStuff/scratch/e2e_real_vllm_to_maxtext_training.py"
```

---

## 4. Full Layer-by-Layer Comparison Table (24 MoE Layers on TPU v5p-8)

### Comparison 1: Inference Logits vs Training Logits (WITHOUT Router Replay)

| Layer / Output | Max Abs Error ($L_\infty$) | Mean Abs Error (MAE) | Cosine Similarity | Alignment Status |
|---|---|---|---|---|
| **Layer 0 - 23 (All 24 Layers)** | `0.000000e+00` | `0.000000e+00` | `1.000000` | **ALIGNED (0.0 Error)** |
| **Final Output Logits** | `0.000000e+00` | `0.000000e+00` | `1.000000` | **ALIGNED (0.0 Error)** |

---

### Comparison 2: Inference Logits vs Training Logits (WITH Real vLLM Router Replay)

| Layer | Max Abs Error ($L_\infty$) | Mean Abs Error (MAE) | Cosine Similarity | Status |
|---|---|---|---|---|
| **Layer 0** | `1.401663e-03` | `2.672930e-04` | `1.000000` | ROUTED SHIFT |
| **Layer 1** | `5.880356e-03` | `1.570321e-03` | `0.999999` | ROUTED SHIFT |
| **Layer 2** | `1.140237e-02` | `2.525885e-03` | `0.999997` | ROUTED SHIFT |
| **Layer 3** | `1.785254e-02` | `3.459606e-03` | `0.999996` | ROUTED SHIFT |
| **Layer 4** | `1.988935e-02` | `4.416998e-03` | `0.999994` | ROUTED SHIFT |
| **Layer 5** | `2.567889e-02` | `5.274020e-03` | `0.999993` | ROUTED SHIFT |
| **Layer 6** | `2.797091e-02` | `6.205670e-03` | `0.999992` | ROUTED SHIFT |
| **Layer 7** | `3.085208e-02` | `7.101245e-03` | `0.999991` | ROUTED SHIFT |
| **Layer 8** | `3.401971e-02` | `7.969804e-03` | `0.999991` | ROUTED SHIFT |
| **Layer 9** | `3.831005e-02` | `8.703190e-03` | `0.999989` | ROUTED SHIFT |
| **Layer 10** | `4.594207e-02` | `9.405111e-03` | `0.999989` | ROUTED SHIFT |
| **Layer 11** | `4.869652e-02` | `9.832682e-03` | `0.999990` | ROUTED SHIFT |
| **Layer 12** | `5.164194e-02` | `1.026941e-02` | `0.999989` | ROUTED SHIFT |
| **Layer 13** | `5.219698e-02` | `1.102508e-02` | `0.999988` | ROUTED SHIFT |
| **Layer 14** | `4.833317e-02` | `1.153090e-02` | `0.999989` | ROUTED SHIFT |
| **Layer 15** | `5.027932e-02` | `1.178815e-02` | `0.999989` | ROUTED SHIFT |
| **Layer 16** | `5.746341e-02` | `1.266397e-02` | `0.999988` | ROUTED SHIFT |
| **Layer 17** | `5.978990e-02` | `1.335402e-02` | `0.999987` | ROUTED SHIFT |
| **Layer 18** | `6.645083e-02` | `1.334405e-02` | `0.999986` | ROUTED SHIFT |
| **Layer 19** | `7.417238e-02` | `1.381461e-02` | `0.999986` | ROUTED SHIFT |
| **Layer 20** | `7.618845e-02` | `1.497370e-02` | `0.999984` | ROUTED SHIFT |
| **Layer 21** | `7.862604e-02` | `1.600818e-02` | `0.999982` | ROUTED SHIFT |
| **Layer 22** | `6.750435e-02` | `1.752816e-02` | `0.999981` | ROUTED SHIFT |
| **Layer 23** | `6.305620e-02` | `1.707263e-02` | `0.999982` | ROUTED SHIFT |
| **Final Logits** | `4.155135e-02` | `5.062298e-03` | **`0.999980`** | ROUTED SHIFT |

---

### Top-1 Token Prediction Agreement
- **Inference Logits vs Training WITHOUT Router Replay**: **100.00%**
- **Inference Logits vs Training WITH Router Replay**: **100.00%**
- **Training WITH Replay vs Training WITHOUT Replay**: **100.00%**

---

## 5. Key Conclusions

1. **Inference vs Training Baseline**: MaxText model inference and natural training mode match with **0.000000 error across all 24 layers and final logits**.
2. **Real vLLM Router Replay**: Passing real vLLM engine router selections into training preserves **100.00% Top-1 Token Prediction Agreement** with high cosine similarity ($> 0.99998$) across all 24 layers.
