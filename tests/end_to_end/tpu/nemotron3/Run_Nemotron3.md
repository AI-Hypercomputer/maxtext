<!--
 # Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# Nemotron-3

[Nemotron-3-Nano-Omni-30B-A3B](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) is NVIDIA's
hybrid Mamba-2 / MoE / attention model. MaxText supports its text backbone (`model_name=nemotron3-nano-omni-30b-a3b`,
`decoder_block: nemotron3`); the vision and audio encoders are not yet supported.

- 52 layers set by `hybrid_override_pattern`: 23 Mamba-2, 23 MoE and 6 grouped-query attention layers without
  positional embeddings (NoPE).
- Each MoE layer has 128 routed non-gated two-matrix `relu2` experts (top-6, sigmoid scores plus a float32
  `e_score_correction_bias`) and one 3712-wide shared expert, written as `shared_experts: 2` with
  `base_moe_mlp_dim: 1856`.
- Only `scan_layers=false` is supported: the layer pattern does not repeat.

**Please note:**

- The router bias must stay float32 in checkpoints and in the model. The 128 biases of a layer sit within about 0.1 of
  each other, so bfloat16 rounding changes the top-6 choice.
- The TPU configurations below are the ones that were tested, not requirements.

## Checkpoint conversion

[1_test_nemotron3.sh](30b-a3b/1_test_nemotron3.sh) converts the Hugging Face checkpoint to an unscanned MaxText
checkpoint on CPU (`scan_layers=false lazy_load_tensors=true`):

```sh
export HF_TOKEN=<your token>            # or HF_MODEL_PATH=/path/to/local/hf_checkpoint
export BASE_OUTPUT_PATH=gs://your-bucket/nemotron3
bash tests/end_to_end/tpu/nemotron3/30b-a3b/1_test_nemotron3.sh
# Output: ${BASE_OUTPUT_PATH}/unscanned/0/items
```

## End-to-end test

[2_test_nemotron3.sh](30b-a3b/2_test_nemotron3.sh) runs the stages below on TPU. Each stage is toggled by
`RUN_<STAGE>=true|false` and prints `E2E_STAGE_<name>_RC=<rc>`. Point `UNSCANNED_CKPT_PATH` at the converted checkpoint
and set `BASE_OUTPUT_PATH`, `HF_TOKEN` and `DATASET_PATH` (C4 for pre-training).

| Stage        | Default | What it checks                                                                             | Main knobs                                                                                        |
| ------------ | ------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `LOGIT`      | on      | fp32 logits against the short golden with router replay (below)                            | `LOGIT_MATMUL_PRECISION` (default `high`), `LOGIT_ICI_FSDP`, `LOGIT_ICI_EP`, `GOLDEN_LOGITS_PATH` |
| `LOGIT_LONG` | off     | fp32 and bf16 logits on 129 / 1025 / 4096-token goldens                                    | `GOLDEN_LONG_DIR`, `LOGIT_LONG_TAGS`, `LOGIT_LONG_DTYPES`                                         |
| `DECODE`     | on      | greedy fp32 decode starts with the reference continuation (`autoregressive_decode_assert`) | `DECODE_CASES` (TSV `prompt<TAB>expected`), `DECODE_ICI_EP`                                       |
| `PRETRAIN`   | on      | 20 steps on C4; the final loss must be finite and positive                                 | `PRETRAIN_ICI_EP` (default 8), `PRETRAIN_MAX_TARGET` (4096), `PRETRAIN_STEPS`                     |
| `FINETUNE`   | on      | 10 SFT steps on `HuggingFaceH4/ultrachat_200k`; the final loss must be finite and positive | `SFT_ICI_EP` (default 4), `SFT_MAX_TARGET` (1024), `SFT_EXTRA_FLAGS`                              |

### Logit check and router replay

The `LOGIT` stage uses [nemotron3_logit_check.py](../../../utils/nemotron3_logit_check.py), which runs the same
full-model forward as `forward_pass_logit_checker` with the fp32 flags from the
[model bring-up guide](../../../../docs/guides/model_bringup.md) (`dtype=float32 weight_dtype=float32 activations_in_float32=true float32_logits=true`). Free routing alone is not a reliable measure for this model: the
router biases of a layer sit within about 0.1 of each other, so exact fp32 top-6 ties occur on real text, and one
flipped expert choice changes the Mamba state for every later token. The checker therefore also runs a replay pass:
the reference top-k is replayed in every MoE layer (`forced_routed_experts`) and MaxText's own top-k is recorded
(`return_routed_experts=true`). The run fails if MaxText's own choice differs from the reference at any cell that is
not a near-tie (reference rank-6/7 margin at most 2 fp32 ulp of the layer's largest absolute bias), so replay only overrides
genuine ties.

Gates:

- fp32 replay: no disagreement outside near-ties, max KL \<= 1e-4 (`--max_kl_div=1e-4` in the
  [model bring-up guide](../../../../docs/guides/model_bringup.md)), mean KL < 1e-3 and max KL \<= 1e-2
  (`.agents/skills/model-bringup/references/logits_checker.md`).
- bf16 replay (`LOGIT_LONG`, `--gate=d_p3_10`): per prompt, mean and max KL(reference fp32 || MaxText) at most 1.25x
  the same statistic of KL(reference fp32 || reference bf16).
- The free-routing pass is reported only.

### Golden

The short golden is `tests/assets/golden_logits/golden_data_nemotron3-nano-omni-30b-a3b.jsonl`; when it is not in the
checkout, the script fetches `gs://maxtext-test-assets/golden_data_nemotron3-nano-omni-30b-a3b.jsonl`. It holds 3
prompts with full-vocabulary fp32 logits, the reference top-k of every MoE layer (fp32 and bf16 routing) and 16-token
greedy continuations. The bf16 gate needs bf16 reference logits, which only the long goldens in `GOLDEN_LONG_DIR`
carry. The goldens were generated
with a patched Hugging Face reference (the generator is kept with the bring-up harness, not in this repository):
`tests/assets/logits_generation/generate_hf_golden_logits.py` runs the unpatched Hugging Face CPU code, whose chunked
Mamba-2 scan is wrong past token 128 for this model.

## Tested topologies and results

All runs were single-slice TPU v5p:

| Stage                                         | Topology | Parallelism and key flags                                                                                                        | Result                                                                                                                                                               |
| --------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logit (fp32, `matmul_precision=high`, replay) | v5p-8    | `ici_fsdp_parallelism=-1 ici_expert_parallelism=1`                                                                               | max KL 7.4820e-06 / 9.3683e-06 / 3.5474e-06, mean KL 4.1517e-06 / 4.8052e-06 / 1.5167e-06 over the 3 prompts; top-1 agreement 1.0; 0 disagreements outside near-ties |
| Decode (fp32)                                 | v5p-8    | `ici_fsdp_parallelism=1 ici_expert_parallelism=-1`                                                                               | 3/3 prompts match the reference continuation                                                                                                                         |
| SFT                                           | v5p-8    | `ici_expert_parallelism=4 optimizer_memory_host_offload=True save_checkpoint_on_start=false save_checkpoint_on_completion=false` | 10 steps, all losses finite: step 0 1.141, minimum 0.713 (step 6), step 9 1.271; median step time 4.0135 s (steps 2-9)                                               |
| Pre-training                                  | v5p-32   | `ici_expert_parallelism=8`                                                                                                       | 20 steps, all losses finite: first 12.238, minimum 9.191, last 9.203; median step time 2.7485 s, 29.973 TFLOP/s/device (steps 2-19)                                  |
| Pre-training                                  | v5p-64   | `ici_expert_parallelism=8`                                                                                                       | 20 steps, all losses finite: first 12.253, minimum 9.126, last 9.126; median step time 2.8365 s, 29.045 TFLOP/s/device (steps 2-19)                                  |

Notes:

- Training was run with expert parallelism 4 on v5p-8 and 8 on v5p-32/64 (the script defaults); other settings were
  not tested.
- SFT on v5p-8 needs `optimizer_memory_host_offload=True`. With it, the step-0 checkpoint save fails, so the script
  sets `save_checkpoint_on_start=false save_checkpoint_on_completion=false` and keeps `enable_checkpointing=true` so
  `load_parameters_path` still restores.
- `attention=flash` needs the sequence length to be a multiple of 128; the logit and decode stages use
  `attention=dot_product`.
