# Phi-4-mini-instruct bring-up

Source: https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json
and the repository's modeling_phi3.py. Model key: `phi4-mini-instruct`.

## Architecture findings

32 homogeneous Phi3 decoder blocks; minimal representative subset: one layer.
Hidden size 3072, SwiGLU intermediate size 8192, 24 query / 8 KV heads,
128 dimensions per head. RMSNorm epsilon 1e-5, no projection biases or dropout.
Tied 200064-token embedding/output matrix; no embedding-logit normalization.
QKV and gate/up weights are fused in HF and split in MaxText.
Partial LongRoPE rotates the first 96 dimensions with split-half layout (no
Llama 3.1 interleaving permutation). Theta 10000, original context 4096,
extended context 131072. Short/long factors are copied from the published config.
Cos/sin amplitude is sqrt(1 + log(32)/log(4096)).

## Phase status

1. Discovery: DONE — real one-layer subset created and HF references generated.
2. Layers: DONE — reused Llama decoder; partial LongRoPE, padding-aware regime selection, and float32 tied-head precision verified.
3. Conversion: DONE — full import in both layouts and full scanned HF export succeed; mini exports match all original tensors.
4. Validation: DONE — 50 regression tests pass; mini/full logits pass in both layouts; 16-token cached mini and full generation match HF exactly; all 194 full-export tensors match the originals.

## Pitfalls and resolutions

- Query scaling is folded into imported query weights, matching the Llama decoder.
- Tied output logits must disable MaxText's default embedding-logit normalization.
- LongRoPE chooses frequencies by maximum actual position, not padded length.
- Crossing the short/long boundary during cached decode changes frequencies for
  new keys only, following HF cache behavior.
- Default system Python lacks JAX; use `../venv1/bin/python` from the MaxText root.
- Sandboxed JAX exposes only CPU. Run TPU commands outside the sandbox; the stalled
  initial sandbox probe held the TPU lock and was stopped. Four TPU devices are usable.
- The tied-head helper unconditionally cast weights to BF16. It now uses config.dtype,
  so FP32 reference tests preserve exact copied weights.
- Orbax 0.12.4 does not recognize the converter's LazyTensor leaves; use
  `--lazy_load_tensors=false`. No generic checkpointing changes were needed.
- Full export exhausted the remaining shared-memory capacity (other VM data already
  occupied most of it). The task-created incomplete export was removed; full export
  succeeded at `/tmp/phi4-full-scanned-hf`, where disk capacity was sufficient.
- A truncated model has highly peaked logits. The checker needs
  `--clip_logits_epsilon=1e-30` to avoid infinite KL from FP32 softmax underflow.
- MaxText decode emits one prefill token plus max_target_length minus
  max_prefill_predict_length autoregressive tokens. Use lengths 19 and 4 for 16 tokens.
- Mini conversion requires override_model_config=true. Mini export also needs
  --override_model_architecture=true.

## Reproduction and validation

See [the usage guide](docs/guides/phi4-mini.md) for portable conversion/check/decode commands.
This session uses Python `../venv1/bin/python`, JAX 0.11.1, Transformers 5.16.0.dev0,
Orbax 0.12.4, and HF revision `cfbefacb99257ffa30c83adab238a50856ac3083`.

Paths on this VM:
- HF full: `/dev/shm/hengtaoguo_google_com/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083`
- HF mini: `/dev/shm/hf_mini/phi4-mini-instruct_1layers`
- Orbax: `/dev/shm/hengtaoguo_google_com/phi4-{mini,full}-{scanned,unscanned}/0/items`
- Mini HF re-exports: the matching Orbax root with `-hf` appended.
- Full scanned HF re-export: `/tmp/phi4-full-scanned-hf`.
- Golden logits: `/dev/shm/hengtaoguo_google_com/phi4-{mini,full}-golden.pkl`
- Golden decode IDs/text: `/dev/shm/hengtaoguo_google_com/phi4-{mini,full}-decode.json`
- Validation logs: `/tmp/phi-{check,decode,convert,export}-{mini,full}-{scanned,unscanned}.log`

CPU regression command (passed):
```bash
JAX_PLATFORMS=cpu PYTHONPATH=src ../venv1/bin/python -m pytest \
  tests/unit/phi4_layers_test.py tests/unit/embeddings_test.py \
  tests/unit/partial_rotary_embedding_test.py tests/unit/param_mapping_test.py -q
```

Session reproduction helpers are `/tmp/prepare_phi4.py` (download/slicing) and
`/tmp/validate_phi4.py` (invokes the central MaxText tools; no custom converter).
Their persistent equivalents and full commands are documented below.

### Recreate the real mini subset

```bash
export HF_HOME=/dev/shm/$USER
python3 - <<'PY'
import json, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
source = Path(snapshot_download(
    'microsoft/Phi-4-mini-instruct', revision='cfbefacb99257ffa30c83adab238a50856ac3083',
    allow_patterns=['*.json', '*.safetensors', '*.jinja']))
target = Path('/dev/shm/hf_mini/phi4-mini-instruct_1layers')
target.mkdir(parents=True, exist_ok=True)
for f in source.glob('*.json'):
    if not f.name.endswith('.index.json'):
        shutil.copyfile(f, target / f.name)
for f in source.glob('*.jinja'):
    shutil.copyfile(f, target / f.name)
config = json.loads((target / 'config.json').read_text())
config['num_hidden_layers'] = 1
config.pop('auto_map', None)
(target / 'config.json').write_text(json.dumps(config, indent=2))
weights = {}
for shard in source.glob('*.safetensors'):
    with safe_open(shard, framework='pt', device='cpu') as f:
        for key in f.keys():
            if key.startswith('model.layers.0.') or key in ('model.embed_tokens.weight', 'model.norm.weight'):
                weights[key] = f.get_tensor(key)
save_file(weights, str(target / 'model.safetensors'), metadata={'format': 'pt'})
print(source)
PY
```

### Convert both layouts and generate references

```bash
export HF_HOME=/dev/shm/$USER
export PHI_HF=/dev/shm/hf_mini/phi4-mini-instruct_1layers
for scan in false true; do
  python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml model_name=phi4-mini-instruct \
    base_num_decoder_layers=1 override_model_config=true scan_layers="$scan" \
    base_output_directory="$HF_HOME/phi4-mini-$scan" hardware=cpu \
    skip_jax_distributed_system=true --hf_model_path="$PHI_HF" \
    --lazy_load_tensors=false --save_dtype=float32 --simulated_cpu_devices_count=1
 done

python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
  --model-id="$PHI_HF" --hf-model-path="$PHI_HF" --trust-remote-code=false --hf-load-dtype=float32 \
  --output-format=pickle --output-path="$HF_HOME/phi4-mini-golden.pkl" \
  --prompts='I love to;The capital of France is;<|user|>What is 2 + 2?<|end|><|assistant|>'

python3 - <<'PY'
import os, torch
from transformers import AutoTokenizer, Phi3ForCausalLM
path = os.environ['PHI_HF']
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
model = Phi3ForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation='eager').eval()
inputs = tokenizer('I love to', return_tensors='pt')
with torch.no_grad():
    result = model.generate(**inputs, do_sample=False, max_new_tokens=16, min_new_tokens=16)
ids = result[0, inputs['input_ids'].shape[1]:].tolist()
print(ids)
print(repr(tokenizer.decode(ids)))
PY
```

### Mini logits, cached generation, and export

```bash
for scan in false true; do
  python3 -m tests.utils.forward_pass_logit_checker \
    src/maxtext/configs/base.yml model_name=phi4-mini-instruct \
    base_num_decoder_layers=1 override_model_config=true scan_layers="$scan" \
    tokenizer_path="$PHI_HF" load_parameters_path="$HF_HOME/phi4-mini-$scan/0/items" \
    base_output_directory="$HF_HOME/phi4-validation" run_name=phi4-logits \
    per_device_batch_size=1 max_target_length=32 max_prefill_predict_length=16 \
    dtype=float32 weight_dtype=float32 activations_in_float32=true \
    matmul_precision=highest float32_logits=true float32_qk_product=true \
    attention=dot_product --golden_logits_path="$HF_HOME/phi4-mini-golden.pkl" \
    --max_kl_div=0.001 --clip_logits_epsilon=1e-30

  python3 -m maxtext.inference.decode \
    src/maxtext/configs/base.yml model_name=phi4-mini-instruct \
    base_num_decoder_layers=1 override_model_config=true scan_layers="$scan" \
    tokenizer_path="$PHI_HF" load_parameters_path="$HF_HOME/phi4-mini-$scan/0/items" \
    base_output_directory="$HF_HOME/phi4-validation" run_name=phi4-decode \
    per_device_batch_size=1 max_prefill_predict_length=4 max_target_length=19 \
    dtype=float32 weight_dtype=float32 matmul_precision=highest \
    float32_logits=true float32_qk_product=true attention=dot_product \
    decode_sampling_strategy=greedy prompt='I love to' --verbosity=1

  JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_huggingface \
    src/maxtext/configs/base.yml model_name=phi4-mini-instruct \
    base_num_decoder_layers=1 override_model_config=true scan_layers="$scan" \
    load_parameters_path="$HF_HOME/phi4-mini-$scan/0/items" \
    base_output_directory="$HF_HOME/phi4-mini-$scan-hf" hardware=cpu \
    skip_jax_distributed_system=true weight_dtype=float32 \
    --hf_model_path="$PHI_HF" --override_model_architecture=true
 done
```

For full-model reproduction, point `PHI_HF` at the printed full HF snapshot,
remove `base_num_decoder_layers=1 override_model_config=true`, and use `full`
in place of `mini` in checkpoint/golden paths. Full export does not require
`--override_model_architecture=true`.

## Measured results

All logits checks used the three prompts in the commands above, all vocabulary
entries, float32 weights/activations, dot-product attention, and the unchanged
KL acceptance threshold of 1e-3. The checker used its 1e-30 probability floor.

| Checkpoint | Maximum token KL across all prompts |
| --- | ---: |
| One layer, unscanned | 6.14045e-5 |
| One layer, scanned | 6.14398e-5 |
| Full model, unscanned | 8.88318e-5 |
| Full model, scanned | 7.65938e-5 |

Full-model cached greedy generation matched HF token-for-token and text-for-text:

```text
[1729, 7187, 13, 23353, 7187, 382, 261, 2212, 2006, 316, 4484, 620, 3283, 326, 8400, 634]
" read books. Reading books is a great way to learn new things and improve your"
```

The initial unscanned mini run emitted 17 copies of token 35428 (` definitions`);
its first 16 matched the HF reference. The scanned mini check used lengths 4/19 and matched all 16 HF token IDs and the complete decoded text.

Regression suite: 50 passed, plus seven subtests. After adding the padded-boundary
model assertion, the Phi-only suite again passed all four tests and seven subtests.
Pyink and `git diff --check` pass. The broader Pylint error-only check reports an
existing `Attention.qk_rope_head_dim` no-member issue outside the changed code.
The new Phi test file passes the Pylint error-only check. `mdformat` is not installed in this environment.

This validates numerical bring-up and cached generation on four TPU devices.
Full 131072-token inference, performance tuning, downstream benchmarks such as
MMLU, LoRA, and quantized checkpoint parity were not evaluated.

Full scanned HF export audit: all 194 tensors match the original pretrained shards
with `rtol=2e-7, atol=1e-7`; maximum absolute error is `5.960464477539063e-08`.
