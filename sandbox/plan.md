
Code branch: https://github.com/AI-Hypercomputer/maxtext/compare/shuningjin-emb

1. We use the folder `sandbox` for temporary files

2. do not commit change 

3. use tpu-testing skill

4. pyink
- read the file changed in https://github.com/AI-Hypercomputer/maxtext/compare/shuningjin-emb
- run commands like
```
pre-commit run --files \
check_qwix_interception.py \
docs/reference/core_concepts/quantization.md \
src/maxtext/configs/base.yml \
src/maxtext/configs/types.py \
src/maxtext/layers/quantizations.py \
tests/unit/quantizations_test.py
```

5. Test plan

(a)
```
python sandbox/check_qwix_interception.py
```
save the output to sandbox/check_qwix_interception_log.txt, only retain the line contains the word "DEBUG" or "[QWIX]"


(b) unit test
```
JAX_PLATFORMS=cpu pytest tests/unit/quantizations_test.py -k "LogitsProjQwixInterceptionTest"
```

6. Rationale: Quantizing Logits Projection for Both Main and MTP

- **Shared Head Architecture**: In DeepSeek-V3 and MaxText, MTP does not define a separate vocabulary projection layer. It explicitly reuses the main model's output head (`mtp_logits = self.decoder.apply_output_head(...)`). Targeting `decoder/logits_dense.*` naturally intercepts both invocations:
  - `op=dot_general0`: Main model forward logits projection.
  - `op=dot_general1`: MTP block forward logits projection.
- **Multiplied FLOP Savings**: The vocabulary projection GEMM is executed $(1 + M)$ times per forward pass (where $M$ is `mtp_num_layers`). Quantizing `logits_proj` cuts the arithmetic cost of this heavy GEMM by $\approx 2\times$ across the main pass and all MTP layers.
- **Gradient & Numerical Parity**: The single shared weight tensor receives gradients from both main and MTP heads during training. Quantizing both paths ensures consistent numerical representation and prevents gradient mismatch between FP8 and BF16 regimes.
- **Speculative Decoding Alignment**: At inference time, if MTP is used for speculative draft generation, having both heads in FP8 matches latency profiles and numerical precision between draft and target tokens.

---

## 7. Prior art: Megatron-Core / Megatron-Bridge

All statements below verified against `NVIDIA/Megatron-LM@main` and
`NVIDIA-NeMo/Megatron-Bridge@main` source, fetched 2026-09-21.

### 7.1 Default is unquantized, by construction

`megatron/core/models/gpt/gpt_model.py:293-297`:

```python
output_layer_cls = (
    TELMHeadColumnParallelLinear
    if is_mxfp8_output_proj_active(config)
    else tensor_parallel.ColumnParallelLinear
)
```

The fallback `tensor_parallel.ColumnParallelLinear` is Megatron's own module, not a
Transformer Engine module. FP8 in Megatron comes from TE modules executing inside the
`fp8_autocast` region, so a non-TE linear stays BF16 **even with `--fp8` enabled**.
This is a mechanism, not a policy: no TE module means no FP8 GEMM.

`TransformerConfig.fp8_output_proj` defaults to `False`
(`megatron/core/transformer/transformer_config.py:685`).

### 7.2 Opt-in is gated to MXFP8 only

`transformer_config.py` (~line 1825):

```python
if self.fp8_output_proj:
    if not self.fp8:
        raise ValueError("fp8_output_proj must be used together with fp8 mode.")
    if self.fp8_recipe != Fp8Recipe.mxfp8:
        raise ValueError("fp8_output_proj requires fp8_recipe='mxfp8', ...")
```

Tensorwise (current-scaling), delayed-scaling, and blockwise recipes **cannot** quantize
the head at all. Only MXFP8 (E8M0 scale per 32-element block) is permitted.

`is_mxfp8_output_proj_active` (`megatron/core/fp8_utils.py:748-763`) additionally requires
Transformer Engine to be installed.

### 7.3 What NVIDIA ships for DeepSeek-V3 671B

Scanned every `src/megatron/bridge/perf_recipes/deepseek/*/deepseek_v3.py`:

| Recipe | Precision | `fp8_output_proj` |
| --- | --- | --- |
| `..._1024gpu_h100_bf16`, `..._256gpu_h100_bf16_32nodes` (reference recipes) | BF16 | n/a — FP8 lines commented out |
| `..._1024gpu_h100_fp8cs` | FP8 tensorwise | off (forbidden by validator) |
| `..._1024gpu_h100_fp8sc` | FP8 blockwise | off (forbidden by validator) |
| `..._256gpu_b200_fp8cs` | FP8 tensorwise | off (forbidden by validator) |
| `..._256gpu_b200_fp8mx` | MXFP8 | **off** (helper not called) |
| `..._256gpu_gb200_fp8mx` | MXFP8 | **on** |
| `..._256gpu_b300_fp8mx` | MXFP8 | **on** (+ `fp8_dot_product_attention=True`) |
| `..._256gpu_gb300_fp8mx` | MXFP8 | **on** (+ FP8 DPA) |
| `..._64gpu_gb300_fp8mx_fsdp`, `..._128gpu_gb300_fp8mx_hsdp` | MXFP8 | **on** |
| `..._128gpu_vr200_fp8mx` | MXFP8 | **off** |
| NVFP4 variants | NVFP4 | off |

Enabled via `_enable_deepseek_full_iteration_mxfp8(cfg, fp8_output_proj=True)`
(`perf_recipes/deepseek/common.py:48-76`, which sets `cfg.model.fp8_output_proj`), or
directly as `cfg.model.fp8_output_proj = True` in the GB300 FSDP/HSDP recipes.

### 7.4 Implication for this change

NVIDIA quantizes the DSv3 output projection **only under block-scaled MXFP8 on
Blackwell-class hardware**, and the config system actively refuses it under per-tensor
FP8. Plausible mechanism: per-32-element block scales bound the dynamic range that the
head's outlier-heavy activations must share; per-tensor absmax does not.

MaxText's `fp8_full` rule uses `absmax` calibration with no tile size, i.e. closer to the
regime Megatron forbids for the head.

> OPEN QUESTION: confirm Qwix's default quantization granularity when no tile size is
> supplied (per-tensor vs per-channel). This determines how closely `quantize_*` here
> matches NVIDIA's MXFP8 recipe, and how strong the accuracy risk is.

---

## 8. Motivation and benefit

### 8.1 It is one of the largest single GEMMs left unquantized

Head FLOPs per token = `2 * d_model * V`. Once `decoder/.*layers.*` and `mtp_block/.*`
are FP8, the vocabulary projection is one of the few remaining BF16 matmuls, so it caps
the achievable end-to-end speedup (Amdahl).

The share depends strongly on model shape:

| Model | `d_model` | `V` | head params | active params | head share of fwd FLOPs |
| --- | --- | --- | --- | --- | --- |
| deepseek3-671b | 7168 | 129280 | 0.93B | ~37B | ~2.5% |
| dense 8B | 4096 | 128256 | 0.53B | 8B | ~6.6% |
| dense ~1B | 2048 | 128256 | 0.26B | ~1.2B | ~20% |

Head cost scales with `V`, not with depth. For DeepSeek-scale MoE it is a ~2.5% forward
slice; at 2x FP8 MXU throughput that is ~1.2% of forward time. Real, not dramatic.

### 8.2 MTP multiplies it (the DeepSeek-specific argument)

MTP has no output head of its own — `multi_token_prediction.py:532` calls
`decoder.apply_output_head`. The head GEMM therefore runs `(1 + M)` times per forward,
where `M = mtp_num_layers`. With `mtp_num_layers=1` the DSv3 head share roughly doubles
to ~5%. This is the strongest quantitative argument for the flag on DeepSeek-V3 and is
the reason the single flag is worth more here than on a comparable dense model.

### 8.3 Train/serve numerical parity

FP8 serving stacks commonly quantize `lm_head`: both shipped `qwen3.5-*-fp8.yml` configs
map it, and `integration/vllm/weight_converter.py:408` maps
`base.decoder.logits_dense.kernel -> vllm_model.lm_head.weight`. If training keeps the
head in BF16 and serving applies PTQ to it, the quantization error is absorbed entirely
at deploy time with no training-time adaptation. Training with it quantized lets the
weights adapt to the FP8 grid.

### 8.4 What it does NOT buy

- **Not weight HBM.** Qwix quantizes dynamically at the matmul; the master weight stays
  in `weight_dtype`. Weight storage is the separate `unquantized_modules` /
  `weight_dtype=float8_e4m3fn` mechanism.
- **Not logits activation memory.** The output is `B*T*V` in accumulation precision
  regardless, and `cast_logits_to_fp32` still applies afterwards.
- Savings are MXU time on 3 GEMMs (fwd, dgrad, wgrad), plus possibly an FP8
  saved-for-backward input of shape `B*T*d`.

### 8.5 Why the default must stay `False`

The output head feeds softmax/cross-entropy directly, its activations carry large
outliers, and its gradient is the origin of the whole backward chain. The DeepSeek-V3
FP8 recipe deliberately keeps the embedding, output head, MoE gate, normalization, and
attention operators in BF16/FP32. Combined with 7.4 (NVIDIA forbids this under
per-tensor FP8), the honest framing of this flag is: an opt-in knob to measure the
speed/accuracy tradeoff on a component the standard recipe deliberately excludes — not
a default-on optimization.

---

## 9. Note on shared (tied) embeddings

`quantize_*` is rejected at config init when `logits_via_embedding=True`. Reasons and
outlook:

### 9.1 Why it is rejected today

With tied embeddings there is no `logits_dense` module at all —
`nnx_decoders.py:467` only constructs it under `if not config.logits_via_embedding`.
The tied path instead calls `attend_on_embedding` (`embeddings.py:188-193`), a bare
`jnp.dot` against the transposed embedding table, executed inside `Decoder` and **not**
inside the `token_embedder` module scope. Qwix rules key on module path, so there is no
addressable path for a rule: the enclosing scope is just `decoder`, and a rule there
would over-match. Without the rejection the flag would be a silent no-op.

### 9.2 Tying is not a storage conflict

Worth stating explicitly, because it is the objection people reach for first and it does
not apply here. Qwix quantizes dynamically at the op, so the tied table remains
BF16/FP32 in HBM and in the checkpoint; the FP8 operands are transient. There is no
"one tensor, two dtypes" problem of the kind weight-only/static quantization would hit.

### 9.3 The two real obstacles

1. **Gradient contamination of the lookup path.** Untied, FP8 wgrad error lands on a
   tensor with a single consumer. Tied, that tensor *is* the input embedding, so E5M2
   backward error perturbs the token representations entering layer 0 on the next step.
   This is a new error channel, not merely more of the same. Magnitude is unmeasured.
2. **Scaling granularity fits embedding tables badly.** Row norms are
   frequency-dependent and span a wide range. Per-tensor absmax over a `V x d` table is
   set by the largest row, so rare-token rows underflow in E4M3. Wider spread than a
   typical weight matrix. Block-scaled (MX-style) quantization mitigates this; see the
   open question in 7.4.

### 9.4 The payoff ranking is inverted

Tied models are exactly the small-`d_model`, large-`V` models where the head dominates:

| Model | `d_model` | `V` | head share of params |
| --- | --- | --- | --- |
| Gemma 3 1B (tied) | 1152 | 262144 | ~302M of ~1B → **~30%** |
| Llama 3.2 1B (tied) | 2048 | 128256 | ~262M of ~1.2B → ~22% |
| deepseek3-671b (untied) | 7168 | 129280 | 0.93B of ~37B active → ~2.5% |

So the currently supported case has the least to gain and the unsupported case the most.

### 9.5 Megatron does not support it either

`gpt_model.py:310-311` passes
`skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights`,
and `TELMHeadColumnParallelLinear.__init__`
(`megatron/core/extensions/transformer_engine.py:1932-1933`) raises
`ValueError("TE output projection does not support skip_weight_param_allocation.")`.
Tied + `fp8_output_proj` therefore hard-fails whenever the head stage also holds the
embedding (PP=1).

> NOTE: under PP>1 with embedding and head on separate stages that guard is not tripped.
> Whether the combination actually works there is UNVERIFIED — do not cite this as
> "Megatron forbids tied" without that qualifier.

### 9.6 If support is added later

Prerequisites, in order:

1. Wrap the tied attend in a named submodule (e.g. `decoder/logits_attend`) so a path
   regex can target it. This refactor, not the quantization, is the actual work.
2. Block-scaled quantization available in the MaxText Qwix path, or confirmation that
   the default granularity is already sub-tensor (7.4).
3. A concrete tied model actually being trained in FP8. Absent that, this is speculative
   surface area.

Safer first increment when it happens: quantize forward and dgrad but keep **wgrad** in
high precision for the tied rule. The shared table then never receives
quantization-noised gradients (removes obstacle 9.3.1 entirely) while retaining roughly
two-thirds of the FLOP saving.

Implementation detail to remember: `attend_on_embedding` hardcodes
`jnp.asarray(embedding_table, jnp.bfloat16)`, so in the tied path `logits_dot_in_fp32`
only sets `preferred_element_type` (accumulation), not the operand dtype. The untied
rationale for rejecting `logits_dot_in_fp32` does **not** transfer verbatim.