# HuggingFace Model Discovery & Architecture Analysis

This guide covers Phase 1 of the MaxText model bring-up workflow: identifying the model, inspecting its architecture/codebase, analyzing decoder patterns, and preparing a minimal safetensors subset for rapid iteration.

## 1. Locate the HF Model ID
- Identify the exact model name on HuggingFace Hub (e.g., `Qwen/Qwen2.5-7B`, `google/gemma-2-9b`, `THUDM/glm-4-9b-chat`).
- The model ID is typically formatted as `{organization}/{model_name}`.

## 2. Download & Inspect HF Source Code
> [!IMPORTANT]
> **Avoid Filling Up TPU VM local disk space**:
> Always set the environment variable `export HF_HOME=/dev/shm/$USER` before running python scripts or commands that download, load, or interact with HuggingFace transformers models.

To implement JAX layers and do parameter copy verification, you must understand the original PyTorch/HF implementation.
- **Option A (Python Inspection)**: Use python to dynamically inspect the configuration and module:
  ```python
  from transformers import AutoConfig, AutoModelForCausalLM
  config = AutoConfig.from_pretrained("HuggingFace/Model-ID", trust_remote_code=True)
  print(config)
  ```
- **Option B (Source Code Inspection)**: Search the `huggingface/transformers` repository or cached files to examine the model implementation class (usually in `src/transformers/models/{model_name}/modeling_{model_name}.py`).

## 3. Key Architectural Components & Decoder Pattern Analysis
Examine the model's PyTorch class (typically subclassing `nn.Module`) and identify:

1. **Decoder Layer Pattern (Homogeneous vs Inhomogeneous)**:
   - **Homogeneous**: All decoder layers are identical and repeated (e.g., standard LLaMA/Gemma). Smallest test subset: **1 layer** ($N=1$).
   - **Dense-to-MoE Transition**: Early layers are dense (e.g., layers 0, 1, 2) and subsequent layers are MoE. Smallest test subset: **First $M$ dense layers + 1 MoE layer** (e.g., 3 + 1 = 4 layers).
   - **Cyclic Attention Patterns**: Alternating attention mechanisms (e.g., local sliding window $\\to$ local $\\to$ global attention; or linear attention with periodic full softmax attention). Smallest test subset: **1 full cycle** (e.g., 4 layers).

2. **Layer Normalization style**: RMSNorm vs LayerNorm, custom scaling, epsilon values.
3. **Attention Block style**:
   - Is it Multi-Head Attention (MHA), Grouped-Query Attention (GQA), or Multi-Query Attention (MQA)?
   - Does it use RoPE (Rotary Position Embeddings)? What are its configuration parameters (e.g., base timescale, partial rotary factor, theta)?
   - Does it use customized attention layers (e.g., Linear Attention like Gated Delta Net)?
   - **Autoregressive Support**: Note how the attention module handles KV caching and causal masks during step-by-step autoregressive generation.
4. **MLP Block style**:
   - Gated MLPs (SwiGLU, GeGLU) vs standard MLP.
   - Mixture of Experts (MoE) parameters: Routing gate, number of experts, active experts per token, shared vs routed experts.

---

## 4. Prepare Mini HF Safetensors Subset in `/dev/shm/hf_mini`

To drastically speed up iteration, save memory, and eliminate large checkpoint download/conversion bottlenecks, create a minimal safetensors model containing only the smallest layer subset ($N$ layers):

1. Target Directory: `/dev/shm/hf_mini/{model_name}_{N}layers`
2. Extract and save:
   - `config.json` with `num_hidden_layers` (or `num_layers`) set to $N$.
   - Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.).
   - Model weights containing ONLY:
     - Token embeddings (`embed_tokens.weight`)
     - Layers $0$ through $N-1$ (`layers.0.*` through `layers.{N-1}.*`)
     - Final normalization (`norm.weight` / `model.norm.weight`)
     - Output language model head (`lm_head.weight` if untied)
3. Use real pretrained weights for this mini subset.
4. This mini HF model will be used in Phase 3 for fast `to_maxtext` conversion and in Phase 4 for both `forward_pass_logit_checker` and 16-token autoregressive decoding comparison.

```python
# Quick snippet to slice HF model to mini safetensors subset
import os
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

model_id = "{hf_model_id}"
out_dir = f"/dev/shm/hf_mini/{model_name}_{N}layers"
os.makedirs(out_dir, exist_ok=True)

# 1. Save modified config
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config.num_hidden_layers = N  # minimal representative subset
config.save_pretrained(out_dir)

# 2. Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(out_dir)

# 3. Slice and save safetensors weights (layers 0..N-1, embed_tokens, norm, lm_head)
# Save sliced state_dict to out_dir/model.safetensors
```

---

## 5. Handover Document Update

At the end of Phase 1, create and populate `{model_name}_handover.md` in the MaxText root directory:
- Document all HF architectural parameters, epsilon, theta, dimensions, and decoder pattern.
- Note the chosen minimal layer count ($N$) and mini safetensors directory path.
- Mark Phase 1 as `DONE` and Phase 2 as `IN PROGRESS`.
