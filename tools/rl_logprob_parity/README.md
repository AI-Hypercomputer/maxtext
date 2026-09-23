# RL Logprob Parity, Prefix Caching & Mamba Concurrency Microbenchmarks

This directory contains microbenchmarking and reproduction tools to evaluate log-probability parity and state integrity between the **MaxText trainer** (teacher-forced forward pass in Flax NNX) and the **vLLM / tpu-inference rollout sampler** (Qwen3.5-35B-A3B hybrid Mamba/Transformer on Cloud TPU v5p/v7x).

---

## 1. Overview of Tools

1. **`microbench_prefix_caching.py`**:
   - Single-turn cold vs. warm prefix caching microbenchmark.
   - Compares exact per-token logprobs between vLLM sampler and MaxText teacher-forced trainer.
   - Tests aligned vs. unaligned prefix lengths.

2. **`microbench_multiturn.py`**:
   - Sequential multi-turn benchmark using real Tunix agentic chat parsers and message history.
   - Tests cache eviction across conversations under `custom_mamba_cache_multiplier=1` (33 Mamba blocks pool).
   - Demonstrates that purely sequential eviction falls back cleanly to prefill recomputation.

3. **`repro_concurrency_eviction.py`**:
   - **Concurrent Multi-Turn Eviction & Contamination Reproducer**:
     - Configured with `max_num_seqs=2`, `num_requests=8`, `ep=8` (`dp_size=4` ranks, 2 requests/rank), and `custom_mamba_cache_multiplier=1` (5 Mamba blocks per rank).
     - Submits 8 concurrent multi-turn requests that share a common prefix (mirroring GRPO / SWE group rollouts).
     - Round 1 consumes initial pool capacity (2 blocks/req * 2 reqs = 4 blocks <= 5 blocks).
     - Round 2 and Round 3 expand sequence lengths to 3+ blocks (2 reqs * 3 blocks = 6 blocks > 5 blocks), forcing Mamba block eviction and slot recycling under continuous concurrent batching.

---

## 2. Empirical Findings: Mamba Concurrency & Cache Eviction

### Reproduction Results (`repro_concurrency_eviction.py`)
Run on **8x TPU v7x** with `Qwen3.5-35B-A3B` on `atwigg/mlperf`:
- **Round 1 (Cold Prefill, 8 concurrent requests)**: All 8 requests generated coherent, correct initial responses.
- **Round 2 (First Continuation)**: Handled cleanly.
- **Round 3 (Concurrent Eviction Triggered)**:
  When sequence lengths exceeded pool capacity (needing 6 blocks out of 5 available per rank), dynamic slot recycling across continuous batching caused **immediate recurrent state contamination**:
  - **Conversation 0 (Topic: Python file sorter)**: Contaminated with Task #4 (process memory segments). Emitted multiple spurious `<|im_end|>` and `</think>` tags mid-sentence:
    ```
    I cannot show the `git diff` because I do not have access to your local Git repository...
    However, I can you<|im_end|>
    </think>
    1<|im_end|>
    </think>
    1 memory segments** (text, data, bss, heap,
    ```
  - **Conversation 1 (Topic: Raft consensus)**: Contaminated with Task #5 (SQL window functions). Grammar collapsed and emitted broken tags and foreign topic:
    ```
    I cannot execute commands on<|im_end|>
    </think>
    I cannot access to have access to your local Git repository, file system state persist the context of the code changes you made to pass the tests/p> I can provide the<|im_end|>
    </think>
    **Task #5: Write an SQL query with window functions
    ```
  - **Conversation 3**: Generated multiple consecutive closing think tags:
    ```
    <think>

    </think>

    </think>

    </think>
    ```

### Root Cause Analysis
1. **The Bug**:
   In `tpu-inference` under `mamba_cache_mode = "align"`, GDN linear attention checkpoints recurrent state once per forward pass at `(seq_len - 1) // mamba_block_size`. Intermediate blocks allocated during continuous batching or slots recycled during cache eviction still contain stale / uninitialized memory.
2. **Missing Fix in `atwigg/mlperf`**:
   The `atwigg/mlperf` branch does not have the fix from `origin/wxd/fix-async-mamba-prefix-race` (commits `85620a006` and `940db7da7`):
   - Commit `85620a006`: Implements read-suppression via `_written_mamba_slots` tracking and masks out `has_initial_state` when reading an unwritten/recycled slot, preventing unwritten slot contamination.
   - Commit `940db7da7`: Reverts the written-boundary clamp that desynchronized MambaManager bookkeeping.
3. **Impact on Teacher-Forcing & RL Training**:
   When the rollout sampler produces contaminated text (e.g. Conversation 1 generating SQL tokens instead of Raft tokens), the MaxText trainer evaluates $\log p_{\text{trainer}}(y)$ conditioned on the true Raft prompt. Because predicting foreign SQL tokens from a Raft prompt has near-zero probability ($\log p \le -30$), the multiplicative probability error $\exp(|\Delta \log p|)$ explodes into the billions, triggering 100% trajectory rejection.

---

## 3. How to Run the Reproduction

```bash
CONDA_PY=/mnt/disks/persist/vllm_conda/bin/python

PYTHONPATH=/home/wenxindong_google_com/maxtext:/home/wenxindong_google_com/tunix:/home/wenxindong_google_com/tpu-inference \
$CONDA_PY tools/rl_logprob_parity/repro_concurrency_eviction.py \
    --max-num-seqs=2 \
    --num-requests=8 \
    --custom-mamba-cache-multiplier=1 \
    --ep=8 \
    --context-repeat=5 \
    --out-dir="/mnt/disks/persist/concurrency_repro"
```

Results and conversation traces are dumped to `concurrency_repro_results.json`.
