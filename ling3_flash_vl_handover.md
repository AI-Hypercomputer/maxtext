# Ling-3.0-flash-VL text-only bring-up

## Status: six-layer mini validation complete

Validated on 2026-09-16. The previous one-layer handover is superseded: one
KDA+dense layer cannot validate this hybrid architecture, and the previously
listed artifacts were absent.

| Phase | Status |
| --- | --- |
| 1. Architecture discovery and real-weight subset | DONE |
| 2. Text decoder implementation and layer tests | DONE for mini scope |
| 3. Central checkpoint conversion | DONE: scanned and unscanned mini |
| 4. Logits and 16-token cached decoding | DONE for mini scope |
| Full 42-layer checkpoint validation | NOT RUN |

## Minimal representative subset

The MLA path now reuses MaxText `AttentionOp` for QK products, softmax/value
aggregation, and normalization. Ling3 retains its projections, interleaved
RoPE, output gate, position/segment mask, and cache layout; checkpoint parameter
names and shapes are unchanged. This uses the shared dot-product arithmetic,
not flash-kernel dispatch or the shared MLA compressed-cache implementation.
KDA retains its feature-wise gated recurrence, which differs from GatedDeltaNet.

Use the **first six layers (0–5)**, retaining the original hidden dimensions,
vocabulary, and all experts:

| Layers | Attention | MLP |
| --- | --- | --- |
| 0–1 | Kimi Delta Attention (KDA) | Dense SwiGLU |
| 2–4 | KDA | 512 routed experts, top 8, plus one shared expert |
| 5 | Gated multi-head latent attention (MLA) | Same MoE |

Six is the minimal contiguous prefix that covers the attention cycle and dense
transition. Selecting fewer experts changes routing and is not equivalent.
Late-layer SwiGLU clipping is covered separately by small tests at layers 34,
35, and 40; it is not exercised by the real-weight prefix.

Keep `model.word_embeddings.weight`, `model.layers.0.*` through
`model.layers.5.*`, `model.norm.weight`, and `lm_head.weight`. Exclude the vision
encoder, multimodal projector, and any MTP weights. The subset contains **6,256
HF tensors**, mapping exactly once to **124 MaxText parameter leaves** in each
layout. Every source tensor shape was checked against the central shape map.

## Source and architecture

Checkpoint: [inclusionAI/Ling-3.0-flash-VL config](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL/blob/554184d95863a873051e8d8ccdb08c32c2c53a03/config.json),
revision `554184d95863a873051e8d8ccdb08c32c2c53a03`.

The VL release lacks its advertised `modeling_bailing_moe_v3_vl.py`. The
reference uses the publisher's [text-backbone decoder](https://huggingface.co/inclusionAI/Ling-3.0-flash/blob/e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3/modeling_bailing_moe_v3.py),
revision `e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3`, loaded with the VL text config
and VL checkpoint weights. `hf_reference.py` retains the upstream decoder,
MLA, router, MLP, and norms. It replaces CUDA-only FLA convolution, gated norm,
and KDA kernels with plain PyTorch CPU formulas. It also adapts config and
RoPE construction to the installed Transformers version. This is an adapted
HF-reference comparison, not an unmodified CUDA HF execution.

- Hidden size 2,560; vocabulary 157,184; 42 layers.
- KDA: 32 heads × 128 dimensions; bias-free causal convolution width 4;
  query/key L2 epsilon `1e-6`; decay
  `exp(-5 * sigmoid(exp(A_log) * (f(x) + dt_bias)))`;
  head-wise RMSNorm with sigmoid output gate.
- MLA: KV rank 512; 128 nonrotary + 64 rotary query/key dimensions;
  value width 128; interleaved text RoPE, theta 6,000,000;
  attention scale `1/sqrt(192)`; head-wise sigmoid output gate.
- All RMSNorm epsilons: `1e-6`.
- Dense intermediate size 6,144; routed/shared intermediate size 768.
- MoE: sigmoid routing, 8 groups/top 4 groups, 512 experts/top 8 experts;
  correction bias affects selection only; selected original scores normalize
  to routing scale 2.5; one shared expert.
- Configured late-layer limits: routed SwiGLU 4 for layers 35–41;
  shared SwiGLU 5 for layers 35–39 and 7 for layers 40–41. Apply the upper
  clamp **after SiLU**, and clamp the up branch symmetrically.

## Results

All numerical validation used FP32 parameters and activations.

| Check | Result |
| --- | --- |
| Small layer, cache, mapping, clipping tests | 5 tests passed (multiple layer subcases) |
| Unscanned central logits checker, 3 prompts | Maximum KL `3.28565e-7` |
| Scanned central logits checker, 3 prompts | Maximum KL `1.61208e-7` |
| TPU cached prefill against reference | Maximum KL `9.66277e-8` |
| TPU cached greedy decoding | 16/16 generated IDs exactly match |
| Standard `maxtext.inference.decode`, CPU | Same 16-token output; exit 0 |
| Real tensor mapping audit | 6,256/6,256 keys and shapes, both layouts |

Logits threshold: maximum per-token KL `< 1e-3`. Small negative KL values in
FP32 logs are numerical roundoff. Prompts: `I love to`, `The capital of France
is`, and `Explain why the sky looks blue in one short sentence.`

Prompt tokens: `[40, 2318, 297]`. Generated tokens on both sides:

```text
[10986, 20506, 42525, 4572, 64403, 79880, 89800, 21092,
 3318, 3318, 19907, 54454, 201, 8609, 20477, 17992]
```

Decoded text, with the carriage return escaped:

```text
I love to fuel多元榄idedamaczek劲的Through介介っ一回\rrumEYOO
```

## Changes and resolved issues

- Replaced nonexistent MLA/MoE class references with concrete Ling text modules.
- Removed KDA convolution biases absent from the source checkpoint.
- Corrected KDA L2 epsilon and made recurrent/convolution caches ignore padding
  and reset at packed segment boundaries.
- Added gated MLA with the correct rotary width and interleaved RoPE. Tests
  include positions above 1,024.
- Added grouped sigmoid routing, expert correction bias, and late-layer clipping.
- Added central `PARAM_MAPPING`, bidirectional `HOOK_FNS`, `HF_SHAPE`, and HF
  config registration. Linear weights transpose; convolution axes reverse;
  decay vectors reshape; experts stack along their explicit expert axis.
- Added per-leaf cache metadata and MaxEngine insertion support. Both direct
  cached TPU execution and the standard decode entry point were verified.
- Explicit FP32 precision is necessary in TPU router/attention matmuls; default
  precision initially changed routing enough to exceed the KL threshold.
- This checkout's Orbax v1 rejects the converter's `LazyTensor` wrapper. Use
  `--lazy_load_tensors=false`; eager central conversion passed.

## Scope and limitations

Full-model configuration defaults to `scan_layers=false`. Scanned mode is
explicitly restricted to **one six-layer cycle**: repeating the first cycle
would incorrectly repeat its dense prefix and would lose late-layer clipping
changes. The full 42-layer model, BF16 parity, training performance, and HF
export/reload have not been validated. Vision/video and MTP are outside this
text-only bring-up. The JAX recurrence and expert implementation prioritize
correctness; no throughput claim is made.

## Artifacts

- Mini: `/dev/shm/hf_mini/ling3_flash_vl_6layers`
- Pinned HF source and hashes: mini's `reference/` and `provenance.json`
- Unscanned Orbax: `/dev/shm/hengtaoguo_google_com/ling3-mini-unscanned/0/items`
- Scanned Orbax: `/dev/shm/hengtaoguo_google_com/ling3-mini-scanned/0/items`
- Goldens: `/tmp/ling3-golden.jsonl` and `/tmp/ling3-golden.jsonl.decode.json`
- TPU result: `/tmp/ling3-golden.jsonl.maxtext.json`
- Logs: `/tmp/ling3-{unit,audit,golden,convert,convert-scanned,checker,checker-scanned,decode,standard-decode}.log`

Shared-memory artifacts are ephemeral. Reusable scripts are checked in under
`tests/assets/ling3/`. No weights are added to the repository.

## Reproduction commands

Run from the MaxText root on the TPU VM. In the agent sandbox, host `/dev/shm`
and TPU execution require escalated execution.

```bash
export HF_HOME=/dev/shm/$USER
export PYTHONPATH=src
LING_PYTHON=../venv1/bin/python
LING_MINI=/dev/shm/hf_mini/ling3_flash_vl_6layers

# HTTP-range slicing; retains all source experts and downloads pinned reference code.
"$LING_PYTHON" -m tests.assets.ling3.prepare_mini --output="$LING_MINI"

JAX_PLATFORMS=cpu LING3_REFERENCE_DIR="$LING_MINI/reference" \
  "$LING_PYTHON" -m unittest tests.unit.ling3_layers_test
JAX_PLATFORMS=cpu "$LING_PYTHON" -m tests.assets.ling3.validate_mini audit --mini="$LING_MINI"
"$LING_PYTHON" -m tests.assets.ling3.validate_mini golden --mini="$LING_MINI" \
  --output=/tmp/ling3-golden.jsonl
```

Convert and check both layouts using the central frameworks:

```bash
for LING_LAYOUT in unscanned scanned; do
  LING_SCAN=false
  if [ "$LING_LAYOUT" = scanned ]; then LING_SCAN=true; fi
  LING_OUTPUT="$HF_HOME/ling3-mini-$LING_LAYOUT"
  JAX_PLATFORMS=cpu "$LING_PYTHON" -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml model_name=ling3-flash-vl \
    base_num_decoder_layers=6 override_model_config=true scan_layers="$LING_SCAN" \
    hardware=cpu skip_jax_distributed_system=true per_device_batch_size=1 \
    max_target_length=32 max_prefill_predict_length=16 dtype=float32 weight_dtype=float32 \
    base_output_directory="$LING_OUTPUT" --hf_model_path="$LING_MINI" \
    --save_dtype=float32 --simulated_cpu_devices_count=1 --lazy_load_tensors=false

  JAX_PLATFORMS=cpu "$LING_PYTHON" -m tests.utils.forward_pass_logit_checker \
    src/maxtext/configs/base.yml model_name=ling3-flash-vl \
    base_num_decoder_layers=6 override_model_config=true scan_layers="$LING_SCAN" \
    hardware=cpu skip_jax_distributed_system=true per_device_batch_size=1 \
    max_target_length=16 max_prefill_predict_length=8 dtype=float32 weight_dtype=float32 \
    matmul_precision=float32 float32_logits=true float32_qk_product=true \
    activations_in_float32=true tokenizer_path="$LING_MINI" \
    load_parameters_path="$LING_OUTPUT/0/items" \
    --golden_logits_path=/tmp/ling3-golden.jsonl --max_kl_div=0.001
done
```

Verify real cached decoding on TPU and through the standard CPU entry point:

```bash
JAX_PLATFORMS=tpu "$LING_PYTHON" -m tests.assets.ling3.validate_mini decode \
  --mini="$LING_MINI" --output=/tmp/ling3-golden.jsonl \
  --checkpoint="$HF_HOME/ling3-mini-unscanned/0/items"

JAX_PLATFORMS=cpu "$LING_PYTHON" -m maxtext.inference.decode \
  src/maxtext/configs/base.yml model_name=ling3-flash-vl \
  base_num_decoder_layers=6 override_model_config=true scan_layers=false \
  hardware=cpu skip_jax_distributed_system=true per_device_batch_size=1 \
  max_target_length=23 max_prefill_predict_length=8 dtype=float32 weight_dtype=float32 \
  matmul_precision=float32 float32_logits=true float32_qk_product=true \
  activations_in_float32=true tokenizer_path="$LING_MINI" tokenizer_type=huggingface \
  add_bos=false prompt='I love to' attention=dot_product \
  load_parameters_path="$HF_HOME/ling3-mini-unscanned/0/items"
```

## Full-model expert sharding correction

The full-model `jit_create_sharded_state` HBM failure exposed replicated routed
expert banks. `Ling3MoE` now annotates wi_0/wi_1 with logical axes
`(exp, embed_moe, mlp_moe)` and wo with `(exp, mlp_moe, embed_moe)`.
Existing MaxText rules map these to expert, FSDP, and tensor parallelism.
Tensor names, shapes, and checkpoint conversion mappings are unchanged.

A shape-only audit of all 42 layers with BF16, TP=8, EP=4, FSDP=1 resolves
all 120 routed expert arrays to 32-way partitioning. Total global weights are
231.743 GiB; resolved per-device parameter storage is 12.322 GiB, including
7.031 GiB routed experts. On v5p-128, TP=8 and EP=8 use all 64 devices and
halve routed-expert storage again. These figures exclude caches and HLO temporaries.
Other custom Ling projections remain replicated. The gather-based expert
implementation is correctness-oriented; this does not establish full-scale
throughput or full decode peak HBM.

Reproduce the audit without allocating full-model weights:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 \
  PYTHONPATH=src ../venv1/bin/python -m tests.assets.ling3.audit_sharding
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  PYTHONPATH=src ../venv1/bin/python -m unittest tests.unit.ling3_sharding_test
```

The sharding test checks expert/tensor placement and numerical parity for
normal routing and forced cross-shard routing, including late-layer clipping.
Four-device CPU and local TPU forward checks passed; all five existing
publisher-reference layer tests also passed. Full GCS checkpoint decoding on
the user's XPK slice has not been rerun with this change. Rebuild the runner
image with the updated source before launching; reconversion is unnecessary.
