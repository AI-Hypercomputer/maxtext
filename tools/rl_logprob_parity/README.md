# MaxText trainer vs vLLM sampler logprob parity (Qwen3.5-35B-A3B / 397B-A17B)

Scripts behind the "Compare logprobs between MaxText trainer and vLLM sampler" tables (TIS band
[0.999, 1.002] statistics). They are experiment scripts, not tests: paths, worktrees and the output
directory (`/mnt/disks/persist/pr4925_repro`) are hard-coded for the host they ran on (see "Environment").
`run_all.sh` holds the exact invocation for every table cell, in dependency order.

## Setup measured
* Trainer: MaxText Qwen3.5 (nnx), fsdp=8, bf16 (fp8 rows: qwix `fp8_full` dynamic quantization).
* Sampler, native: vLLM + tpu-inference torchax path (`MODEL_IMPL_TYPE=vllm`), tp=8 with
  `attn_dp_size=4` (attention DP-4 x TP-2), `--enable-expert-parallel` (EP-8), bf16 KV cache,
  chunked prefill, block 256. FP8 sampler = `Qwen/Qwen3.5-35B-A3B-FP8`.
* Sampler, MaxText-in-vLLM: `maxtext_vllm_adapter` (`MODEL_IMPL_TYPE=flax_nnx`), attention DP-4 x TP-2, MoE TP-8.
* Data: 8 x 512 real tokens (MaxText `docs/*.md`, Qwen3.5 tokenizer), produced by `run_35b_real.py`
  into `real35b_L3.npz` (tokens + layer-3 hidden state); every other script reads that file.

## Table rows -> scripts
| rows | sampler side | trainer side |
|---|---|---|
| 35B prompt tokens, layers 4/8, bf16/bf16 | `torchax_prefix.py` (also captures per-layer expert ids for replay) | `maxtext_prefix.py` |
| 35B prompt tokens, 40 (full), bf16/bf16 and bf16/fp8 (+ fp8 layers 4/8) | `torchax_prefix_gen.py` (`NL`, `MODEL`) | `maxtext_prefix_gen.py` (`NL`, `MODES`, `TX_NPZ`, `SUFFIX`) |
| 35B fp8/fp8 (qwix trainer), 40 (full) | same FP8-sampler npz | `maxtext_prefix_gen.py` with `FULLCALL=1 EXTRA='{"quantization": "fp8_full", "use_qwix_quantization": true}'` |
| 35B output tokens (decode path) | `vllm_generate.py` (rollouts, `enable_return_routed_experts`) | `maxtext_score.py` (`GEN_ONLY_REPLAY=1` = replay row); `compare_output.py` prints the row |
| 35B MaxText-in-vLLM, 40 (full) | `adapter_prompt_logprobs.py` (real engine `prompt_logprobs`) | `maxtext_full_logprobs.py` |
| 397B layers 4/8, bf16/fp8 | `torchax_prefix_397b.py` | `maxtext_prefix_397b.py` (`MODE=own|replay`, 8-layer partial restore) |
| tables | `build_tables.py` renders the markdown tables from the npz outputs | |

Row semantics: layers 4/8 = logit lens (model's own final norm + lm_head on the hidden state after
layer N, both sides); 40 = true logprobs. "Expert replay Y" = the sampler's per-layer top-k expert ids
fed to the trainer's `RoutedMoE.get_topk` (class patch in `maxtext_prefix*.py` / `maxtext_score.py`),
weights recomputed from the trainer's fp32 gate logits. Output-token replay covers decode positions
only: the engine's `routed_experts` rows for prefill positions are zero-filled under attn_dp +
chunked prefill, and `maxtext_score.py` maps those rows (and all prefill rows) to the trainer's own routing.

## Running
`run_all.sh` — steps 0-8; each step needs the whole TPU. Wrappers:
* `run_maxtext.sh <script.py> [args]` — MaxText-side environment (conda + tpu-inference/vLLM/MaxText trees on `PYTHONPATH`).
* `run_vllm.sh <script.py> [args]` — vLLM/tpu-inference environment with the production serving env vars
  (`USE_MOE_EP_KERNEL=0`, `NEW_MODEL_DESIGN=1`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, ...); `ADAPTER=1` switches to
  the MaxText-in-vLLM adapter (`MODEL_IMPL_TYPE=flax_nnx`, MaxText tree on `PYTHONPATH`).
* `wait_tpu.sh <log> <wrapper> <script.py> [args]` — waits for `/dev/vfio/*` to be free, runs, retries if the TPU was taken.

## Environment
A conda env plus `PYTHONPATH` pointing at a tpu-inference worktree (Qwen3.5 canonical weight mapping), a vLLM
worktree and this MaxText tree (`src/`); `HF_HOME` with the HF checkpoints; MaxText-format checkpoints under
`/mnt/disks/persist`. The vLLM engine is started in-process (`vllm.LLM`) so the harness can call the loaded,
sharded model directly.
