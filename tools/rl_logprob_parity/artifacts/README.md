# Run artifacts: 32 real r2e-gym prompts, Qwen3.5-35B-A3B bf16

MaxText trainer vs MaxText-in-vLLM (adapter) sampler, 8 x v7x (`tpu7x-8`), attention DP-4 x TP-2 sampler /
TP-2 trainer, EP off, 32 prompts x 32768 tokens + 8192 sampled tokens each.

Prompts come from `../r2e_prompts_32.jsonl`, decoded from the GBS1024 `run170964` RL trace
(`gs://xiaotongyang-bucket/mlperf-rl-prep/gbs1024_run170964/nemo_logs_170964/exp_001/train_data_step6.jsonl`).

## Contents

| path | what |
|---|---|
| `metrics.txt` | full `--stage compare` output, both sections, per-sequence geomeans |
| `parity_adapter.npz` | the same metrics as arrays (`prompt_*`, `output_*` keys) |
| `generations/index.txt` | one line per rollout: mean logp, length, opening line |
| `generations/row{00..31}.txt` | prompt tail + the 8192 sampled tokens, control tokens line-broken |

## Results

| | tokens | per-token oob | seq `is_oob_ratio` | kept | \|dlogp\| med / p99 / max |
|---|---|---|---|---|---|
| PROMPT (prefill) | 1,048,380 | 98.58% | **84.38%** | 5/32 | 0.0550 / 6.379 / 29.89 |
| OUTPUT (decode) | 262,144 | 97.79% | **93.75%** | 2/32 | 0.0341 / 0.310 / 17.22 |

Band `[0.999, 1.002]`, seq-mask-tis semantics (`nan_to_num` the log ratio, masked mean per sequence,
exponentiate, reject outside the band, report the rejected fraction).

For reference, the same metric computed directly from the trace's own stored `prev_logprobs` /
`generation_logprobs` (Qwen3.5-397B on GB200 under NeMo-RL, 1024 sequences) gives `is_oob_ratio` 55.12%
with per-token 20.47% — and there, 55.47% of sequences fall below the lower bound while none exceed
the upper one.

## Repro

```bash
export PYTHONPATH=/wenxindong/mnt/disks/persist/pydeps_rl_parity
cd <maxtext-worktree>

python -u tools/rl_logprob_parity/compare_trainer_sampler.py \
  --prompts-file tools/rl_logprob_parity/r2e_prompts_32.jsonl \
  --prompt-len 32768 --gen-tokens 8192 \
  --trainer-micro-batch 4 --trainer-tp 2 --gpu-memory-utilization 0.7 \
  --out-dir /wenxindong/mnt/disks/persist/rl_r2e32 \
  --hf-home /wenxindong/mnt/disks/persist
```

Re-score without touching the TPU:

```bash
python tools/rl_logprob_parity/compare_trainer_sampler.py --stage compare \
  --prompt-len 32768 --out-dir /wenxindong/mnt/disks/persist/rl_r2e32
```

Stage timings on 8 x v7x: tokenize 5 s, engine up 80 s, prompt pass 40 s, 8192-token rollouts 192 s,
trainer model load 83 s, 8 chunks of 4 rows at 40960 tokens 92 s, compare 3 s -- about 8.5 minutes.

`--trainer-micro-batch`/`--trainer-tp` are not optional at this length: one forward over all 32 rows
exhausts HBM, and TP shards the vocab dimension of the logits.

## Caveats

* **The rollouts are forced continuations.** `ignore_eos=True` with `min_tokens=8192` means the model
  cannot stop at `<|im_end|>`, so after its first tool call it invents both sides of the conversation.
  Most rows degrade into hallucinated tool responses and then control-token soup (row 7 is a clear
  example; row 1 stays coherent longest). The OUTPUT numbers therefore describe off-distribution
  self-talk, not the on-policy rollouts the RL trainer scores, and every sequence here is maximally
  "overlong" -- exactly what `overlong_filter` would discard in production.
* **164 non-finite log-ratios** appear in the prompt section and are zeroed by `nan_to_num`. The
  synthetic-corpus runs had none. Unexplained; the sampler returned no logprob for those positions.
* Prompts are truncated at exactly 32768 tokens, so each cuts mid-trajectory rather than at a turn
  boundary, and generation resumes from that cut.
* Environment: needs `orbax-checkpoint>=0.12.4`, `qwix>=0.1.8`, `drjax` on `PYTHONPATH`, plus a local
  patch to `maxtext_vllm_adapter/adapter.py` (not committed) because the tpu-inference checkout used
  here predates the 4-argument `_maybe_set_compact_mamba_num_blocks_override`. Neither is needed with
  tpu-inference at main.
* One of three attempts died mid-generation with a TPU `RuntimeUnexpectedCoreHalt`.

Given the first two caveats, treat the 84.38% / 93.75% as provisional.
