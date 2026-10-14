# DeepSeek-V4.1-Flash bring-up handover

Branch `dsv41-flash` (worktree `~/maxtext-dsv41-flash`), based on `hf-stream/6-deepseek-v4`.

## Architecture notes (vs DeepSeek-V4)
- 40 layers, hc_mult=4 hyper-connections with a **delayed pre-mix**: each sub-block's `hc_mixes` produces the
  collapse weights used by the *next* sub-block; the head collapses with the last FFN's `pre` (no HyperHead).
- Every layer: sliding window (128) attention with attn sink. `compress_ratios`: L0-1 = 0, L2-19 = 2, L20-39 = 1.
- Cross-layer sharing: compressed KV + index keys come only from `kv_source_layers` [2,8,14,20]; top-k positions
  only from `index_source_layers` [2,8,14,20,24,28,32,36]; L20 also picks candidate blocks (2048 x 8) that later
  index sources restrict their top-k to.
- No per-head q norm (V4 had one). Compressor: ratio 2 = non-overlapping softmax pooling (fp32), ratio 1 = projection.
  YaRN only on compressed layers. Index keys are derived from the compressor latent (`wk`, `k_norm`).
- Engram (L1, L14): n-gram hash lookup (int64 hash; emulated exactly in uint32 limbs in JAX) gated into each hc copy.
- No hash routing; shared expert is SwiGLU-clamped too (added `activations_limit` to `MlpBlock`).

## Code
- Reference (plain PyTorch, no quant/TileLang): `tests/utils/deepseek_v41_reference/` (`model.py`, `kernel.py`,
  `engram.py`, `runner.py`, `create_mini_checkpoint.py`).
- MaxText: `layers/attention_deepseek41.py`, `layers/engram_deepseek41.py`, `models/deepseek41.py`,
  `nnx_decoders.py` (`_init_deepseek41_layers` / `_apply_deepseek41_layers`), configs
  `deepseek4.1-flash{,-mini}.yml`, `param_mapping.py` (`DEEPSEEK_V41_*`).
- Scanning: layers grouped at every sharing source; consecutive identical groups form `segments_{k}` scanned stacks.
  Mini: [0] [1] [2,3]x2 [6,7] [8,9] [10,11]x2. Full: [0] [1] [2-7]x2 [14-19] [20-23] [24-27]x4.

## Mini checkpoint
`gs://snehalv-data/deepseek-v4.1-flash/hf/mini` (also `/dev/shm/dsv41_mini` on the TPU VM). 14 layers
(orig 0 1 2 3 8 9 14 15 20 21 24 25 28 29), 8 experts, engram_vocab_size 10000, index_topk 4,
candidate_topk_blocks 8, no vision / mtp, unquantized bf16 (+ f32 where upstream is f32).

## Commands
```
# unit + tiny checkpoint tests (CPU)
JAX_PLATFORMS=cpu DEEPSEEK_V41_TOKENIZER=<dir with tokenizer.json> python -m pytest tests/unit/deepseek_v41_vs_reference_test.py
# logit check (TPU v4-8)
python -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml model_name=deepseek4.1-flash-mini \
  run_name=x load_parameters_path=gs://snehalv-data/deepseek-v4.1-flash/hf/mini source_checkpoint_layout=safetensors_dynamic \
  tokenizer_path=<local mini dir> tokenizer_type=huggingface scan_layers={false,true} max_target_length=512 \
  per_device_batch_size=1 dtype=float32 weight_dtype=float32 matmul_precision=highest sparse_matmul=false capacity_factor=-1 \
  skip_jax_distributed_system=true --run_hf_model=true --hf_model_path=<local mini dir> --max_kl_div=0.5
```

## Results
| run | short prompts max KL | ~493-token prompt max / avg KL |
|---|---|---|
| unscanned | 6.1e-5 | 4.7e-2 / 8.3e-4 |
| scanned | 5.3e-5 | 4.7e-2 / 7.8e-4 |

## Known gaps / follow-ups
- Forward / prefill only: no autoregressive decode caches (window ring, compressed KV, compressor state).
- Attention is a dense jnp dot-product path (no flash/splash); fine for bring-up, not for long-context training.
- megablox `sparse_matmul` in fp32 exceeds v4 VMEM with default tiles -> used `sparse_matmul=false` for the check.
- Vision tower, DSpark (mtp.*) and `gate.bias_vl` are not implemented (text only).
- Engram uint32 hash requires bucket primes < 2**24 (true for the real config: ~16.0M).
