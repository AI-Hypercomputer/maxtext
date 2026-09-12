# Phi-4-mini-instruct

Use `model_name=phi4-mini-instruct` for
[`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct).
The model reuses MaxText's Llama decoder with grouped-query attention, SwiGLU,
RMSNorm, tied embeddings, and partial LongRoPE. Both scanned and unscanned
checkpoint conversion are supported, including export back to Hugging Face.

## Configuration

The model has 32 layers, hidden size 3072, intermediate size 8192, 24 query heads,
8 KV heads, and vocabulary size 200064. LongRoPE rotates the first 96 of each
head's 128 dimensions; it uses the published short/long frequency factors and
attention scaling, with a 4096-token original context and 131072-token maximum.
The rotary layout matches HF directly; no Llama-style weight permutation is used.
Padding is excluded when selecting the short/long frequency regime.

The HF tokenizer includes the instruction chat template. For chat inference,
format the prompt with `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)` before passing it to MaxText.

## Convert and decode

Run these commands from the MaxText root with its dependencies installed:

```bash
export HF_HOME=/dev/shm/$USER
python3 -m maxtext.checkpoint_conversion.to_maxtext \
  src/maxtext/configs/base.yml model_name=phi4-mini-instruct \
  base_output_directory="$HF_HOME/phi4-unscanned" scan_layers=false \
  hardware=cpu skip_jax_distributed_system=true \
  --lazy_load_tensors=false --save_dtype=float32

python3 -m maxtext.inference.decode \
  src/maxtext/configs/base.yml model_name=phi4-mini-instruct \
  load_parameters_path="$HF_HOME/phi4-unscanned/0/items" \
  base_output_directory="$HF_HOME/phi4-decode" run_name=phi4-decode \
  scan_layers=false per_device_batch_size=1 \
  max_prefill_predict_length=4 max_target_length=19 \
  dtype=float32 weight_dtype=float32 attention=dot_product \
  matmul_precision=highest float32_logits=true float32_qk_product=true \
  decode_sampling_strategy=greedy prompt="I love to"
```

Use `scan_layers=true` and a separate output directory for a scanned checkpoint.
The eager loading flag above avoids a lazy-tensor handler incompatibility in
Orbax 0.12.4. The converter splits fused QKV and gate/up tensors and folds
`1/sqrt(head_dim)` into the query weights.

To export, use the same `scan_layers` value as the source checkpoint:

```bash
python3 -m maxtext.checkpoint_conversion.to_huggingface \
  src/maxtext/configs/base.yml model_name=phi4-mini-instruct \
  load_parameters_path="$HF_HOME/phi4-unscanned/0/items" \
  base_output_directory="$HF_HOME/phi4-hf" scan_layers=false \
  hardware=cpu skip_jax_distributed_system=true weight_dtype=float32
```

## Verify

```bash
JAX_PLATFORMS=cpu python3 -m pytest \
  tests/unit/phi4_layers_test.py tests/unit/embeddings_test.py \
  tests/unit/partial_rotary_embedding_test.py tests/unit/param_mapping_test.py -q

python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
  --model-id=microsoft/Phi-4-mini-instruct \
  --hf-model-path=microsoft/Phi-4-mini-instruct \
  --trust-remote-code=false --hf-load-dtype=float32 --output-format=pickle \
  --prompts="I love to;The capital of France is" \
  --output-path="$HF_HOME/phi4-golden.pkl"

python3 -m tests.utils.forward_pass_logit_checker \
  src/maxtext/configs/base.yml model_name=phi4-mini-instruct \
  load_parameters_path="$HF_HOME/phi4-unscanned/0/items" scan_layers=false \
  base_output_directory="$HF_HOME/phi4-validation" run_name=phi4-logits \
  per_device_batch_size=1 max_target_length=32 max_prefill_predict_length=16 \
  dtype=float32 weight_dtype=float32 activations_in_float32=true \
  matmul_precision=highest float32_logits=true float32_qk_product=true \
  attention=dot_product --max_kl_div=0.001 \
  --golden_logits_path="$HF_HOME/phi4-golden.pkl"
```

For one-layer bring-up, add `base_num_decoder_layers=1 override_model_config=true`
to MaxText commands and point HF loading/tokenization at a one-layer safetensors
subset. Exporting a subset also requires `--override_model_architecture=true`.
Truncated-model probabilities can underflow during KL computation; the checker's
`--clip_logits_epsilon=1e-30` option avoids this without changing its KL threshold.

LongRoPE switches frequencies when actual positions exceed the original context.
As in HF, already cached keys retain the frequencies used when they were created.
The layer tests cover short/long positions, the transition, unchanged non-rotary
dimensions, padding, fused-weight round trips, and full tiny-model logits in both
layer layouts. Full-context throughput and downstream benchmarks are separate
from numerical bring-up validation.

Bring-up validation on TPU passed for the full model in both layouts (maximum
per-token KL below 9e-5 across three prompts), with exact 16-token greedy decode
agreement against HF. All 194 tensors from a full scanned HF export matched the
original weights within float32 round-off. See the root
`phi4-mini-instruct_handover.md` for measured results and subset reproduction.
