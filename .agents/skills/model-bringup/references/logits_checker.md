# E2E Logits Validation & Autoregressive Decode Verification

This guide covers Phase 4 of the MaxText model bring-up workflow: validating logits, running autoregressive decode checks, and iteratively resolving discrepancies using the fast mini safetensors subset before full model verification.

## 1. Fast Mini-Subset Verification Workflow

Always verify the minimal layer subset ($N$ layers) first using real weights from `/dev/shm/hf_mini/{model_name}_xlayers` and `/dev/shm/$USER/checkpoints/{model_name}_mini_orbax`.

This fast loop consists of two dual validations:
1. **Forward Pass Logits Check**: Verifies prefix prompt logits and probabilities.
2. **Autoregressive Decode Check**: Verifies step-by-step KV-cached generation (16 tokens).

---

## 2. Generate Golden Mini Logits & Golden Decode
Generate standard reference golden logits and golden 16-token autoregressive decode using the local mini PyTorch HuggingFace model in `/dev/shm/hf_mini/{model_name}_xlayers`:

```python
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

user = os.environ.get("USER", "user")
mini_model_path = "/dev/shm/hf_mini/{model_name}_xlayers"
tokenizer = AutoTokenizer.from_pretrained(mini_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(mini_model_path, torch_dtype=torch.float32, trust_remote_code=True)
model.eval()

prompt = "I love to"
inputs = tokenizer(prompt, return_tensors="pt")

# 1. Golden Forward Pass Logits
with torch.no_grad():
  outputs = model(**inputs)
  golden_logits = outputs.logits.detach().cpu().numpy().tolist()

# Save to golden_mini_logits.jsonl for forward_pass_logit_checker
os.makedirs(f"/dev/shm/{user}", exist_ok=True)
with open(f"/dev/shm/{user}/golden_mini_logits.jsonl", "w") as f:
  f.write(json.dumps({"prompt": prompt, "logits": golden_logits}) + "\n")

# 2. Golden 16-Token Autoregressive Decode (Greedy)
with torch.no_grad():
  gen_tokens = model.generate(**inputs, max_new_tokens=16, do_sample=False)
  golden_decoded_text = tokenizer.decode(gen_tokens[0])
  print("HF Golden 16-Token Output:", repr(golden_decoded_text))
  print("HF Golden Token IDs:", gen_tokens[0].tolist())
```

---

## 3. Run MaxText Forward Pass Logits Checker on TPU VM
Execute the logits checker script directly from the MaxText root directory on your TPU VM:

```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.tests.utils.forward_pass_logit_checker   --model_name={model_name}   --maxtext_model_path=/dev/shm/$USER/checkpoints/{model_name}_mini_orbax   --golden_logits_path=/dev/shm/$USER/golden_mini_logits.jsonl   --base_num_decoder_layers={N}
```

### Key Evaluation Metrics:
1. **Average KL Divergence per token** ($D_{KL}(P_{golden} \\parallel Q_{model})$): Expected range $< 10^{-3}$.
2. **Maximum KL Divergence**: Target $\\le 10^{-2}$.
3. **Top-K Token Overlap & Jaccard Similarity**: Confirms top token distributions align.

---

## 4. Run MaxText Autoregressive Decode Check (16 Tokens)
Run greedy decoding on MaxText using the mini converted checkpoint to generate 16 additional tokens:

```bash
export HF_HOME=/dev/shm/$USER

python3 -m maxtext.inference.decode   src/maxtext/configs/base.yml   model_name={model_name}   load_parameters_path=/dev/shm/$USER/checkpoints/{model_name}_mini_orbax   base_num_decoder_layers={N}   prompt="I love to"   max_target_length=20   temperature=0.0
```

### Success Criteria for Decode:
- **Gibberish Match**: Because the model only has $N$ layers (an incomplete model), the output text will naturally be gibberish.
- **The verification passes if MaxText and HuggingFace generate the exact same token IDs and identical gibberish string**.
- This strictly verifies that KV cache management, positional index updates, causal attention masking, and autoregressive step execution in the JAX decoder are bug-free.

---

## 5. Iterative Debugging Loop (Closing the Gap)
If KL divergence is high or decode diverges from token 1:

> [!IMPORTANT]
> **Free Shared Memory Space**:
> If checks fail, immediately delete the converted Orbax checkpoint from `/dev/shm/$USER/checkpoints` using `rm -rf <path_to_checkpoint>` before fixing the bug and running conversion again.

### Step-by-Step Debugging Guide:
1. **Is it off from token 0 / layer 0?**
   - Check token embedder weights, transposition, and input ID mapping.
2. **Is it off in attention / decode step?**
   - Check if RMSNorm epsilon matches HF `config.json`.
   - Verify query/key/value projection transpositions and head dimension layout.
   - Rotary Embeddings (RoPE): Check if RoPE frequency theta matches, and verify whether `permute_to_match_maxtext_rope` is needed.
   - KV Cache: Check if single-step decode attention mask or cache update indexing is misaligned.
3. **MLP & MoE routing**:
   - Check expert gate weights and top-k gating probability normalization (`norm_topk_prob`).

### Deep Verification via Intermediate Activations Mapping:
If discrepancies persist:
1. Run both HF PyTorch and MaxText with $N=1$ layer and feed identical input tokens.
2. Print activations at each sublayer:
   - Input embeddings
   - Pre-attention norm output
   - Attention Q, K, V projections and attention output
   - Post-attention / MLP norm output
   - MLP intermediate and output projections
   - Final norm and logits
3. Compare intermediate tensors side-by-side to pinpoint the exact failing operation.

---

## 6. Final Session Wrap-up & Handover Document
Once the mini subset passes and the full model checkpoint is validated:
1. Finalize `{model_name}_handover.md` in the MaxText root directory.
2. Record:
   - Final status of all 4 phases (`DONE`).
   - Summary of all bugs found during bring-up and their fixes.
   - Complete list of pitfalls and model-specific nuances.
   - Full executable CLI commands for slicing, converting, checking logits, and decoding.
