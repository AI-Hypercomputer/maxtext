# Kimi-K3 GPU reference: text-only, first four layers

## Required result

Run the original Hugging Face text model for `moonshotai/Kimi-K3` on GPU, retaining layers 0–3. Prompt exactly `I love to`, without a chat template, and generate exactly 16 additional greedy tokens. Save token IDs and decoded text. Do not change model math to match MaxText.

Use the same pinned source revision as the existing mini assets:

```text
f831ab66814297da540d832a5235f8e904f29d06
```

This is a fresh reference verification. Earlier reported parity used a CPU short-convolution replacement and is not proof of parity with the original GPU implementation. Do not use the previous output as an expected answer or modify the model to reproduce it.

## Prepare authentic mini assets

The previous mini directory was `/dev/shm/hf_mini/kimi_k3_4layers`. Transfer the packed checkpoint weights and tokenizer assets, but verify the config and custom Python against the pinned original snapshot before using them.

The old `slice_kimi_k3.py` is not suitable unchanged: besides setting four layers, it rewrites `linear_attn_config.full_attn_layers` to `[3, 4]` and `kda_layers` to `[0, 1, 2]`. Do not carry those rewrites into this reference. The released lists use one-based layer numbers; preserve them exactly. Entries beyond layer four can remain, as only four layers are instantiated.

Preparation requirements:

1. Obtain the original config and all custom Python/tokenizer files from the pinned revision. Use the released text backbone configuration and implementation, without the vision model. Preserve the original text config fields, including attention lists, dtype and quantization settings; change only `num_hidden_layers` to `4`. If standalone loading requires text-class registration metadata, record that packaging change explicitly and verify it selects the original text class.
2. Keep the original embeddings, layers 0–3, final normalization, output attention-residual parameters and language-model head. Preserve packed weights and scales byte-for-byte. Removing the multimodal wrapper prefix from tensor names is a packaging operation; do not alter tensor contents or select different layers.
3. Compare the existing mini config against the original text config and report every difference. Remove inherited experimental changes. Compare custom Python files by SHA256 against the pinned snapshot; replace altered files with the originals. Do not accept CPU fallback patches from the previous VM.
4. Validate that all required first-four-layer tensors are present and match the original tensors. Model loading must not leave missing parameters randomly initialized or silently discard required text weights.

Save the config comparison, code hashes, tensor validation and exact source revision alongside the result. Do not modify the old mini directory in place; use a new directory, for example `/dev/shm/hf_mini/kimi_k3_4layers_original_gpu`.

## Runtime requirements

Use the released model's supported CUDA PyTorch, Transformers, compressed-tensors, FLA/flash-linear-attention, Triton and attention dependencies. Follow the pinned model's original dependency/setup instructions. Record exact package versions and GPU details. Do not invent version pins from the old CPU environment.

Provide enough GPU memory for the model's actual loaded/decompressed representation and generation. Packed checkpoint size is not runtime memory usage. Multiple GPUs may be used with ordinary Hugging Face device placement. If memory is insufficient, use a larger GPU VM; do not change precision, quantization, expert counts, attention, layers or kernels to make it fit.

Do not import or install the previous verification adapters:

- `cpu_short_convolution.py`
- `install_lazy_decompression`
- CPU KDA fallbacks or patched model forwards
- Any MaxText comparison or monkeypatch probes

Do not force `attn_implementation="eager"` or substitute convolution, recurrence, normalization, MoE, residual or rounding implementations. Use the original model's GPU execution path and default attention selection. Keep the released weight dtype/quantization. `trust_remote_code=True` permits execution of the original local custom code; it does not change model math or require network access.

## Decode

After asset validation and dependency installation, save this as `decode_kimi_k3_original_gpu.py`. Run from a clean Python process. Set `MINI_DIR` to the newly validated directory and `RESULT_DIR` to a writable output directory.

```python
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mini = Path(os.environ["MINI_DIR"])
out = Path(os.environ.get("RESULT_DIR", "kimi_k3_gpu_reference"))
out.mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available(), "This reference requires CUDA"
config = json.loads((mini / "config.json").read_text())
assert config["num_hidden_layers"] == 4

# Do not request a different attention implementation or weight dtype.
# The validated mini config supplies the original text architecture/settings.
tokenizer = AutoTokenizer.from_pretrained(
    mini, trust_remote_code=True, local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    mini, trust_remote_code=True, local_files_only=True,
    dtype="auto", device_map="auto",
)
model.eval()
assert model.config.num_hidden_layers == 4
# CPU/disk placement would invalidate this GPU-only reference run.
placement = getattr(model, "hf_device_map", {})
assert not any(str(v) in {"cpu", "disk", "meta"} for v in placement.values()), placement
assert all(p.device.type == "cuda" for p in model.parameters())

prompt = "I love to"
inputs = tokenizer(prompt, return_tensors="pt")  # No chat template.
assert inputs["input_ids"][0].tolist() == [40, 3270, 308], "Investigate tokenizer/source mismatch"
inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}
prompt_ids = inputs["input_ids"][0].tolist()
with torch.inference_mode():
    # Neutralize early stopping so exactly 16 new tokens are returned.
    # Greedy selection, repetition penalty and other generation defaults
    # remain as supplied by the original model.
    output = model.generate(
        **inputs, do_sample=False, max_new_tokens=16,
        eos_token_id=None, forced_eos_token_id=None,
        stop_strings=None,
    )
all_ids = output[0].tolist()
new_ids = all_ids[len(prompt_ids):]
assert len(new_ids) == 16, f"Expected 16 new tokens, got {len(new_ids)}"
versions = {}
for name in ("torch", "transformers", "compressed-tensors", "flash-linear-attention", "triton", "accelerate"):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = "not installed under this distribution name"
result = {
    "source_revision": "f831ab66814297da540d832a5235f8e904f29d06",
    "model_class": type(model).__name__,
    "mini_path": str(mini),
    "num_hidden_layers": model.config.num_hidden_layers,
    "prompt": prompt,
    "input_ids": prompt_ids,
    "generated_token_ids": new_ids,
    "full_generated_token_ids": all_ids,
    "continuation_text": tokenizer.decode(new_ids),
    "full_text": tokenizer.decode(all_ids),
    "versions": versions,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "device_map": {k: str(v) for k, v in placement.items()},
}
(out / "hf_decode_original_gpu.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(result, indent=2, ensure_ascii=False))
```

```bash
export MINI_DIR=/dev/shm/hf_mini/kimi_k3_4layers_original_gpu
export RESULT_DIR="$PWD/kimi_k3_gpu_reference"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -u decode_kimi_k3_original_gpu.py 2>&1 | tee kimi_k3_original_gpu.log
```

If the original standalone text class cannot be loaded using AutoModel, inspect its original registration and fix only standalone packaging. If original GPU dependencies fail, resolve installation without replacing kernels. Report any required implementation change instead of making it.

## Return to the MaxText VM

Return `hf_decode_original_gpu.json`, the complete load/decode log, package/GPU information, and the asset-validation records. Preserve the raw result even if it differs from MaxText. Compare generated IDs only after this original GPU reference has completed. A matching decoded string alone is insufficient; all 16 token IDs must match.
