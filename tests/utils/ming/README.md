# Ming bringup validation

Run commands from the repository root. Audio input is currently limited to one file.
The model configuration enables audio; text-only runs must set `use_audio=false use_multimodal=false`.

The HF reference needs PyTorch, transformers, flash-linear-attention, openai-whisper,
torchaudio, and publisher remote code. The CPU adapter replaces GPU KDA kernels
with fla's recurrent mathematical reference and replaces its fused gated RMSNorm.
It also recomputes the nonpersistent RoPE frequencies after loading: with the
transformers 5 environment used for bringup, those buffers were uninitialized.
These substitutions are part of the comparison method, not independent GPU-kernel validation.

## Tests

```bash
export PYTHONPATH=src:.
export JAX_PLATFORMS=cpu
python3 -m pytest -q tests/unit/ming_audio_test.py tests/unit/ming_conversion_test.py
# Explicit path is required: optional HF parity tests are excluded from default discovery.
python3 -m pytest -q tests/unit/ming_layers_test.py
```

The scan tests and audio token tests do not require publisher dependencies. HF
reference tests skip if the remote source or optional dependencies cannot load;
review the skip count before interpreting a run as numerical validation.

## Mini-checkpoint and golden data

Use a representative four-layer cycle (KDA+dense, KDA+MoE, KDA+MoE, MLA+MoE).
The audio encoder, projector, vocabulary and embeddings retain their full size.
Supply your own audio file and ensure sufficient RAM for loading the checkpoint.

```bash
export HF_HOME=/dev/shm/$USER/huggingface
export MING_WORK=/dev/shm/$USER/ming
mkdir -p "$MING_WORK"
python3 -m tests.utils.ming.prepare_mini --output="$MING_WORK/mini" --num_layers=4
python3 -m tests.utils.ming.generate_golden --model_path="$MING_WORK/mini" \
  --audio_path="$AUDIO_PATH" --output="$MING_WORK/golden.pkl" --max_new_tokens=16
mkdir -p "$MING_WORK/tokenizer"
cp "$MING_WORK/mini/"tokenizer*.json "$MING_WORK/mini/special_tokens_map.json" \
  "$MING_WORK/mini/chat_template.jinja" "$MING_WORK/tokenizer/"
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
  model_name=ming-3.0-audio-tiny override_model_config=true base_num_decoder_layers=4 \
  scan_layers=false hardware=cpu skip_jax_distributed_system=true weight_dtype=float32 \
  checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False \
  base_output_directory="$MING_WORK/converted" \
  --hf_model_path="$MING_WORK/mini" --lazy_load_tensors=false --save_dtype=float32 \
  --simulated_cpu_devices_count=1
```

```bash
# Mini logits check (CPU)
JAX_PLATFORMS=cpu python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
  model_name=ming-3.0-audio-tiny override_model_config=true base_num_decoder_layers=4 scan_layers=false \
  hardware=cpu skip_jax_distributed_system=true per_device_batch_size=1 max_target_length=64 \
  max_prefill_predict_length=48 attention=dot_product sparse_matmul=false weight_dtype=float32 dtype=float32 \
  dtype_mm=float32 matmul_precision=highest float32_logits=true float32_qk_product=true \
  tokenizer_path="$MING_WORK/tokenizer" tokenizer_type=huggingface \
  load_parameters_path="$MING_WORK/converted/0/items" \
  --golden_logits_path="$MING_WORK/golden.pkl" --max_kl_div=0.001

# Mini audio+text decode (TPU; unset JAX_PLATFORMS first)
JAX_PLATFORMS=tpu python3 -m maxtext.inference.decode src/maxtext/configs/base.yml model_name=ming-3.0-audio-tiny \
  override_model_config=true base_num_decoder_layers=4 scan_layers=false tokenizer_type=huggingface \
  tokenizer_path="$MING_WORK/tokenizer" load_parameters_path="$MING_WORK/converted/0/items" \
  audio_path="$AUDIO_PATH" prompt='Please answer the question in the audio.' per_device_batch_size=1 \
  max_prefill_predict_length=64 max_target_length=80 attention=dot_product decode_sampling_strategy=greedy \
  weight_dtype=float32 dtype=float32 dtype_mm=float32 matmul_precision=highest float32_logits=true \
  float32_qk_product=true sparse_matmul=false skip_jax_distributed_system=true hardware=tpu

```

Run the logits checker and decode with the converted parameters, the tokenizer-only
directory, the same audio input, fp32, and `sparse_matmul=false`. The tokenizer-only
directory avoids remote-code tokenizer prompts. For full conversion, supply a directory
containing only top-level model shards: recursive loading of the publisher's `talker/`
subdirectory can overwrite thinker weights with identically named tensors.

For scanning, use `scan_layers=true` consistently in conversion and execution.
The 24-layer model uses four unrolled prefix layers and five scanned four-layer blocks;
partial final cycles remain unrolled.

## Historical evidence and current limits

The pre-split handover reported mini logits max KL 2.84e-7 and identical 16-token
fp32 greedy decode; full 24-layer logits max KL 6.14e-8 and identical 64-token fp32
audio decode. Scanned full-model conversion/logits/decode matched the unscanned
results. Default bf16 decode matched the first 32 IDs before numerical drift.
These are historical results, not fresh runs of this PR stack. The checkpoint and
golden artifacts were unavailable in the stack-preparation environment. Rerun real
checkpoint logits and autoregressive decode before treating the split as validated
end to end. A skipped HF-reference test is not a passing parity check.

## Stack preparation checks

The split preserved all original runtime Python ASTs, the model YAML, and pytest
configuration. Reference-test loading was separated from the scan tests, and a
missing-PyTorch regression was fixed with a failing-then-passing test.

Fresh CPU checks during stack preparation: text layer 3 passed / 9 HF skips;
audio layer 10 passed / 13 HF skips; final stack 20 passed / 13 HF skips.
The nine conversion cases cover mapping coverage, expert/block order, optional
audio parameters, and reversible linear/conv transforms in both scan modes.
Pinned Pyink, YAML and spelling checks passed. New implementation/test/utility
files pass pylint; whole-file lint reports existing shared-file findings,
reproduced on the base revision. No shared lint rules were weakened. Publisher
reference parity and real-checkpoint logits/decode remain pending fresh reruns.
