# Kimi-K3 text-only four-layer bring-up

## Scope and status

This experimental implementation targets the first four text layers of `moonshotai/Kimi-K3`: three KDA attention layers followed by MLA, with a dense MLP in layer0 and latent MoE in layers1–3. The preset defaults to four unscanned layers. Scanned and pipeline execution are rejected. Full-model and multimodal behavior are unverified; the full model includes a final MLA layer outside the repeating schedule.

The copied-weight layer/cache tests and mini Orbax conversion have been exercised. **Parity with the unchanged original HF GPU model remains pending.** Historical 16-token equality and central-checker KL0.0003045273 used an HF CPU convolution replacement and must not be treated as acceptance under the original-reference requirement. An extra all-position prefill audit also found KL0.00628728 at one position. Diagnostic tools and generated logit assets have been removed from the PR.

## Original assets

Pinned HF revision: `f831ab66814297da540d832a5235f8e904f29d06`.

The old HF mini was `/dev/shm/hf_mini/kimi_k3_4layers`; it needs validation/reconstruction because the earlier slicing script changed attention-layer lists. The corrected Orbax checkpoint remains at `/dev/shm/hengtaoguo/checkpoints/kimi_k3_mini_fixed/0/items`, with115 parameter leaves. Both old checkpoints are preserved outside the repository. Host shared memory may differ from the sandbox mount.

Follow [the GPU handoff](kimi_k3_gpu_handoff.md) to prepare a text-only mini using original code/config/weights with only `num_hidden_layers=4`. Do not reuse the previous CPU patches. The retained helper uses the original HF GPU path without convolution/decompression overrides or forced eager attention:

```bash
python verification/kimi_k3/decode_hf.py \
  --model-path /dev/shm/hf_mini/kimi_k3_4layers_original_gpu \
  --output /tmp/hf_decode_original_gpu.json
```

Prompt is exactly `I love to`; it requests exactly16 greedy continuation tokens without a chat template. It does not save logits or load MaxText. Validate package versions, source hashes and required checkpoint tensors using the GPU handoff before running it.

## Implementation

KDA stores FP32 recurrent state and convolution histories in NNX caches and ignores prefill padding. MLA adds the released output gate, correct key/value cache widths and combined cached attention softmax. Latent MoE normalizes after experts and selects with biased sigmoid scores while weighting with unbiased normalized scores. Attention residuals reset the prefix at block boundaries. Kimi opts into FP32 dense accumulation followed by BF16 rounding; generic projections keep existing defaults.

Central conversion supports packed MXFP4 weights and includes MLA gating/residual parameters. Lazy checkpoint reads serialize decompression and conversion; single-file and indexed checkpoints retain their loading paths. Router input widths remain unchanged for other models.

Original MaxText decode uses `maxtext.inference.decode`, `tokenizer_type=huggingface`, `tokenizer_trust_remote_code=true`, and a target length19 for this three-token prompt plus16 generated tokens. Custom-code trust is opt-in and defaults to false.

## Validation commands

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src:. python -m pytest \
  tests/unit/kimi_k3_layers_test.py tests/unit/kimi_k3_conversion_test.py \
  tests/unit/linears_test.py tests/unit/moe_test.py::DeepSeekRoutingTest \
  tests/unit/maxengine_tokenizer_test.py -q

JAX_PLATFORMS=cpu PYTHONPATH=src:. \
XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest \
  tests/unit/kimi_k3_layers_test.py -k tensor_parallel_shards -q
```

The central `tests/utils/forward_pass_logit_checker.py` is unchanged. Regenerate golden logits from the original GPU implementation before using it as an acceptance oracle.

Packed MXFP4 conversion requires optional HF dependencies `torch` and `compressed-tensors` with `MXFP4PackedCompressor` support. The conversion environment used compressed-tensors 0.17.0. Install these before converting packed assets; they are not part of the base MaxText install. GPU reference dependencies must follow the pinned original model setup.
