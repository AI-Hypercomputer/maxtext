# Example run: `compare_trainer_sampler.py --gen-tokens 8192`

Qwen3.5-35B-A3B bf16/bf16, MaxText trainer vs MaxText-in-vLLM (adapter) sampler.
8 x v7x (`tpu7x-8`), attention DP-4 x TP-2, MoE TP-8, 8 prompts x 512 tokens, 8192 sampled tokens each.

Engine/JAX/tokenizer noise is stripped; only the script's own output is kept. The compare section was
re-run against the same npz files after the prompt-slicing fix landed, so it reflects the committed code.

```
[    0s] maxtext_root=/home/wenxindong_google_com/maxtext/.claude/worktrees/rl-logprob-unified
[    0s] hf_home=/wenxindong/mnt/disks/persist
[    0s] out_dir=/wenxindong/mnt/disks/persist/rl_logprob_parity_gen8k
[    4s] tokens: (8, 512) from 89 files / 180845 corpus tokens -> .../rl_logprob_parity_gen8k/tokens.npz
[    4s] tokens: first = '```{include} ../CONTRIBUTING.md\n```'
[    4s] === stage: sampler ===
[   80s] adapter engine up
[   85s] sampler: prompt pass done; mean logp=-1.7392 top-1 acc=0.616
[   85s] sampler: generating 8192 tokens x 8 prompts (temperature=1.0)
[  228s] sampler: generation done; mean gen logp=-0.7910
[  228s] sampler: saved .../rl_logprob_parity_gen8k/sampler_adapter_logprobs.npz
[  228s] sampler: engine released
[  228s] === stage: trainer ===
[  228s] trainer: scoring (8, 8704) (prompt 512 + gen 8192)
[  228s] trainer cfg: emb=2048 q=16 kv=2 E=256 k=8 layers=40
[  275s] trainer model loaded
[  333s] trainer: saved .../rl_logprob_parity_gen8k/trainer_logprobs.npz; mean logp=-0.8478 next-token top-1 acc=0.788 (ckpt sanity); mean gen logp=-0.7923

### MaxText trainer vs MaxText-in-vLLM (adapter) sampler - Qwen3.5-35B-A3B bf16/bf16, full model - PROMPT tokens (teacher-forced, prefill)
  tokens=4088  seqs=8  nonfinite(zeroed)=0  band=[0.999, 1.002]
  per-token : oob 71.36%  in-band 28.64%   |dlogp| med 0.0135 / p99 0.255 / max 0.70
   seq  seq_log_is_ratio_mean  seq_geomean_is_ratio  kept
     0          -1.775882e-03              0.998226     0
     1          -3.164294e-03              0.996841     0
     2           7.134458e-03              1.007160     0
     3           5.995312e-03              1.006013     0
     4           6.868934e-04              1.000687     1
     5           2.691010e-04              1.000269     1
     6           1.125078e-03              1.001126     1
     7          -9.984752e-04              0.999002     1
  kept 4/8
  >>> is_oob_ratio (seq-mask-tis) = 0.5000 (50.00%)
      oob_ratio    (per-token)   = 0.7136 (71.36%)

### MaxText trainer vs MaxText-in-vLLM (adapter) sampler - Qwen3.5-35B-A3B bf16/bf16, full model - OUTPUT tokens (sampled, decode)
  tokens=65536  seqs=8  nonfinite(zeroed)=0  band=[0.999, 1.002]
  per-token : oob 50.79%  in-band 49.21%   |dlogp| med 0.0016 / p99 0.214 / max 5.21
   seq  seq_log_is_ratio_mean  seq_geomean_is_ratio  kept
     0          -1.523510e-03              0.998478     0
     1          -1.199224e-03              0.998801     0
     2          -7.731684e-04              0.999227     1
     3          -1.458547e-03              0.998543     0
     4          -7.243200e-04              0.999276     1
     5          -2.218413e-03              0.997784     0
     6          -9.927758e-04              0.999008     1
     7          -1.830136e-03              0.998172     0
  kept 3/8
  >>> is_oob_ratio (seq-mask-tis) = 0.6250 (62.50%)
      oob_ratio    (per-token)   = 0.5079 (50.79%)

saved .../rl_logprob_parity_gen8k/parity_adapter.npz
```

## Notes

* Decode-path per-token agreement is much tighter than prefill (median `|dlogp|` 0.0016 vs 0.0135), yet the
  sequence metric is worse (62.50% vs 50.00% rejected): all eight output geomeans sit below 1.0, a small
  one-sided bias that does not cancel, whereas the prompt geomeans straddle 1.0.
* The trainer scored these 512 prompt positions inside an 8704-token forward pass. Causally they should match
  a prompt-only pass exactly; at bf16 with different splash tiling they do not — a prompt-only run of the same
  prompts gave 68.44% per-token oob against 71.36% here. Do not compare prompt numbers across runs with
  different sequence lengths.
* With 8 sequences the per-sequence metric moves in steps of 12.5%, so small differences in `is_oob_ratio` are
  not resolvable; the per-token figures and the geomean values themselves carry more signal.
* Environment: this run needed `orbax-checkpoint>=0.12.4`, `qwix>=0.1.8` and `drjax` on `PYTHONPATH`, plus a
  local patch to `maxtext_vllm_adapter/adapter.py` because the tpu-inference checkout predated the 4-argument
  `_maybe_set_compact_mamba_num_blocks_override`. Neither is needed with tpu-inference at main.
