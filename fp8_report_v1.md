# FP8 MoE rollout / trainer: GSM8K RL comparison on v5p (Qwen3.5-35B-A3B), v1

Status: **complete**. All three variants finished 30 steps. t16r8 had to be restarted once after an infra event (see
[Issues](#issues-hit-during-the-runs)); the numbers below come from the relaunch.

## TL;DR

Variant names read `t<trainer bits>r<rollout bits>` for the routed experts. The run script, pods and W&B
runs use the older names: t16t16 = `bf16`, t16r8 = `r8`, t8r8 = `r8t8`.

| variant | rollout routed experts | trainer routed experts |
|---|---|---|
| **t16t16** | bf16 | bf16 |
| **t16r8** | FP8 (W8A8) | bf16 |
| **t8r8** | FP8 (W8A8) | bf16 master weights, FP8 fake-quant in the forward pass (straight-through gradients) |

Means over steps 1-29 (step 0 includes compilation, ~730 s for all three):

| variant | step time s | exposed gen s | train s | weight sync s | gen tok/s (step-level) | completion len | `tis/is_oob_ratio` | token logdiff absmean | token logdiff mean | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| t16t16 | 49.2 | 20.9 | 16.1 | 12.2 | 7283 | 370 | 0.572 | 0.0120 | -0.00092 | 0.899 |
| t16r8 | 48.7 | 21.0 | 16.4 | 11.3 | 7686 | 358 | 0.616 | 0.0124 | -0.00112 | 0.894 |
| t8r8 | 55.9 | 27.7 | 16.8 | 11.5 | 6807 | 573 | 0.626 | 0.0178 | -0.00169 | 0.553 |

Findings:

- **The FP8 rollout alone (t16r8) matches t16t16 learning. The fake-quant trainer (t8r8) stalls.**
  - t16t16's reward climbs from 0.61 to ~0.95 within ~6 steps, and its completions shorten from ~490 to ~330 tokens.
  - t16r8 follows the same curve: reward 0.57 → 0.92 by step 5 and ~0.95 after that; completions shorten to ~300-350.
  - t8r8 stays at ~0.55 reward and 510-660 tokens for all 29 steps.
  - All three start from the same point (step 0 reward 0.61 / 0.57 / 0.58).
  - So the stall comes from `fp8_moe_fake_quant` in the trainer, not from the FP8 rollout. Likely suspects: the
    straight-through estimator or the per-step requantization making the effective update to the experts too
    small or noisy at lr 1e-6. Worth checking the expert gradient norms and the weight update after a few steps
    in t8r8 vs t16t16. This is still one seed per variant.
- **FP8 rollout with a bf16 trainer (t16r8) looks ready.** Same reward and step time as t16t16, slightly faster
  weight sync. Its token logdiff is only ~3% above t16t16 on average (0.0124 vs 0.0120). It starts ~15% higher
  (0.0212 vs 0.0187 at step 0) and falls to ~0.008 by the end, below t16t16's ~0.0105.
- **OOB.** With GSM8K's short sequences, `is_oob_ratio` is dominated by noise. Even t16t16 sits at ~0.57, because
  the [0.999, 1.002] band is very tight for ~400-token sequences. t16r8 is 0.616 and t8r8 0.626, both ~0.05 above
  t16t16 (0.572).
- **Token-level mismatch.** This is length-independent, so it is the better signal here.
  - t8r8's `token_logdiff_absmean` is 0.0178 vs 0.0120 for t16t16 (+48%).
  - The mean logdiff bias roughly doubles (-0.0017 vs -0.0009).
  - t16r8, which has the *larger* weight mismatch (FP8 rollout vs bf16 trainer), stays within ~3% of t16t16. So the
    t8r8 gap is mostly not rollout/trainer numerics. It tracks t8r8's trajectories, which stay long and
    low-confidence because the run doesn't learn. Logdiff falls as completions get shorter and more confident.
  - Consistent with that, the runs are much closer at steps 0-2, before they diverge: absmean 0.0187 / 0.0165 /
    0.0184 for t16t16 vs 0.0204 / 0.0181 / 0.0196 for t8r8 (~+10%, similar to t16r8's 0.0212 / 0.0191 / 0.0184).
    The ~10-15% early excess is the real FP8 rollout numerics cost; likely the W8A8 activation quantization.
- **Weight sync** is ~0.7-1 s *faster* with FP8 experts (t16r8 11.3 s, t8r8 11.5 s, t16t16 12.2 s). Halving the
  bytes of the expert tensors saves more time than the on-sync quantization costs.
- **Training time** is about the same (t16t16 16.1 s, t16r8 16.4 s, t8r8 16.8 s). Fake-quant costs little.
- **Step time** is dominated by generation, and generation depends on completion length. t16r8 and t16t16 have
  similar lengths (358 vs 370) and step times (48.7 s vs 49.2 s). t16r8's step-level gen tok/s is ~6% higher
  (7686 vs 7283), which is suggestive but within noise (see the next section). t8r8's completions are ~55% longer,
  so its step time is not a speed comparison.

### Generation speed: what the RL runs can and cannot tell

- **Step-level tok/s** = (completion_length_mean × seqs) / exposed_generation_time.
  - It is confounded by completion length and tail latency: a step waits for the slowest of 256 sequences.
  - Steps 1-2 are where lengths are closest (t16t16 530/506 tokens, t8r8 612/621). There t8r8 is faster:
    2372 / 1924 tok/s vs 2092 / 1436.
  - In steady state (steps 9+) t16t16 generates ~340-token completions at ~10k tok/s and t8r8 ~570-token
    completions at ~8.5k tok/s. The different lengths make these numbers incomparable.
- **Per-request TPOT**, from each trajectory's `inference_metrics.json`, includes queueing time: each replica
  gets 32 concurrent requests with `max_num_seqs=16`. Which variant looks faster depends on the steps chosen:
  - step 10: median 18.3 ms for t8r8 vs 20.4 ms for t16t16;
  - steps 1-12: 38.6 ms vs 22.0 ms.
- **vLLM's own 10 s "Avg generation throughput" windows** are mostly partly idle, because generation lasts only
  10-40 s per step. Too few clean full-batch windows exist to use them.

**Conclusion:** the RL runs do not give a reliable decode-throughput number. The right follow-up is a controlled
benchmark: same prompts, fixed output length, fixed concurrency, one 2x2x1 v5p rollout slice, bf16 vs `fp8_moe`.

## Setup

- **Model and task:** Qwen3.5-35B-A3B on GSM8K, 30 steps, recipe `gsm_8k_35b_256.sh`:
  - GRPO-LOO, batch of 16 prompts × 16 generations;
  - max prompt and response length 1024, lr 1e-6;
  - `seq-mask-tis` with ratio band [0.999, 1.002];
  - seed 42, so all variants see the same prompts at each step.
- **Cluster:** `bodaborg-v5p-nap` (europe-west4), namespace `trellis`. Each variant uses 96 chips:
  - a 64-chip v5p trainer (4x4x4, Pathways, fsdp 32 × tp 2);
  - 8 rollout replicas on v5p 2x2x1 (vLLM, expert parallelism 4);
  - raiden weight sync through the MaxText-to-MaxText weight converter (prefused MoE).
- **FP8 recipe:** only the routed experts (wi_0/wi_1/wo) are FP8.
  - Format float8_e4m3fn, with per-output-channel scales of shape `(E, 1, out)` = amax / 448.
  - All other weights and all FP32 paths are unchanged.
  - The rollout's fused MoE kernel runs W8A8.
- **W&B:** entity `google-trellis`, project `trellis-gsm8k`, runs `igfp8g-{bf16,r8,r8t8}-gsm8k-v5p`.

Per-step metric keys in W&B (all under the `train/` prefix):

- `orchestrator/{step_time_sec,exposed_generation_time,policy_training_time,weight_sync_time}`
- `rollout/{completion_length_mean,global_valid_seqs}`
- `trainer/tis/is_oob_ratio`
- `trainer/sampler_is/token_logdiff_{absmean,mean}`
- `rewards/mean`

## Per-step data

t16t16:

```
 step  step_s  gen_s  train_s  sync_s  gen_tok_s  compl    oob  ld_absmean    ld_mean  reward
    0   727.8  158.3    556.1   13.33      795.1  491.8 0.6068      0.0187  -0.001294  0.6086
    1   94.07  64.88    17.46   11.69       2092  530.2 0.6114      0.0165  -0.001304  0.6102
    2   110.1  90.21    7.665   12.15       1436  505.9 0.6155     0.01841  -0.001477  0.6344
    3   62.97   40.3    11.02   11.59       2659  418.5 0.6061      0.0151   -0.00164  0.8168
    4   63.51  45.57    4.534   13.36       2444    435 0.5103     0.01473  -0.001214  0.7812
    5   61.51  29.71    18.88   12.86       2909  337.7 0.5717     0.01293  -0.001145  0.9437
    6   60.49  31.79    16.23    12.4       3161  392.5 0.5834     0.01157 -0.0006434  0.8758
    7   62.44   31.7    18.69      12       2796  346.2 0.5148     0.01161 -0.0004647  0.9719
    8   59.07  30.49    15.27   13.24       3285  391.3 0.6574     0.01357 -0.0009693  0.8164
    9   39.39  8.829    18.89   11.61      10340  356.8 0.5385     0.01163 -0.0007288  0.9543
   10   39.02  13.23    13.74   11.98       7676  396.8 0.5659     0.01237  -0.001091  0.9094
   11   40.31  10.28    17.21   12.76       9203  369.4 0.6075     0.01118   -0.00114  0.9227
   12   38.89  9.436    17.26   12.13      10020  369.3 0.5522     0.01068 -0.0007389  0.9398
   13   37.08  14.81    10.25   11.96       6535  378.1 0.5988     0.01188 -0.0009019  0.8996
   14   38.55  8.104    18.39   11.99      10470  331.6 0.5273     0.01058 -0.0007165  0.9965
   15   39.37   8.94    17.09   13.28       9409  328.6 0.5491     0.01034 -0.0008809  0.9961
   16    38.7  12.19    13.88   12.56       8161  388.6 0.5641     0.01257     -0.001  0.8344
   17   37.01  7.022    17.55   12.36      10630  291.7 0.5625     0.01071  -0.001087  0.9684
   18   39.93  8.481     18.7   12.69      10490  347.5 0.5768    0.009999 -0.0008507  0.9437
   19   63.28  30.58    20.52   12.12       3279  391.6 0.5663     0.01145 -0.0009315  0.8762
   20   39.07  9.477    16.99   12.54       9851  364.7  0.603     0.01167 -0.0009203  0.9609
   21   57.07  29.75     15.2   12.06       3520  409.1 0.5853     0.01106 -0.0008505  0.9457
   22   38.59  8.351    18.03   12.14      11150  363.6 0.6258     0.01111 -0.0007692  0.8793
   23   37.75  6.784    19.23   11.66      11590  307.1 0.5586     0.01155 -0.0007969  0.9437
   24   38.42  13.31    13.32   11.73       7712  400.9 0.5996     0.01193 -0.0007364  0.8113
   25   37.57  7.201    18.25   12.05      10310  290.1 0.5586     0.01118 -0.0007092  0.9332
   26   38.11  8.224    18.07   11.74      10550  338.8 0.4779     0.01035 -0.0006344   0.968
   27   36.73  7.174    17.43   12.05      10390  291.2 0.6055     0.01026 -0.0007812  0.9965
   28   37.41  8.756    16.96   11.62       9027  308.8 0.5764     0.01052 -0.0008771  0.9785
   29   39.96  8.942    18.59   12.36      10100    353  0.527     0.01057 -0.0006063  0.9695
```

t16r8:

```
 step  step_s  gen_s  train_s  sync_s  gen_tok_s  compl    oob  ld_absmean    ld_mean  reward
    0   734.8  160.2    561.9    12.6      850.9  532.6 0.6628     0.02118  -0.002377  0.5691
    1   87.94  64.58    12.59   10.73       2280  575.1 0.6519     0.01911  -0.001518  0.5453
    2   86.84  66.14    9.436    11.2       1985  512.7 0.6526     0.01836  -0.001611  0.6461
    3   66.21  39.97    15.05   11.13       3029  472.9 0.5975     0.01759  -0.001731  0.7312
    4   64.38  40.28    12.08   11.96       2833  445.7  0.704     0.01845  -0.002001   0.752
    5   81.51  56.46    14.37   10.62       1531  337.6 0.6175     0.01472  -0.001452  0.9191
    6   59.74  30.47    17.94   11.27       3243  386.1 0.6382     0.01467  -0.001137  0.8676
    7   59.08  28.96    18.28   11.79       2990  338.2 0.6221      0.0144 -0.0009491  0.9359
    8   60.95  32.71     15.5   12.68       2972  379.8  0.714     0.01634    -0.0013  0.8254
    9   37.47  12.44    13.87    11.1       7138  346.9 0.6318     0.01372  -0.001029  0.9637
   10   59.26  30.61    16.96   11.64       3146  376.1 0.6626     0.01378  -0.001179  0.8934
   11   36.61  9.269    16.36   10.91       9775  353.9 0.6911     0.01313  -0.001286  0.9367
   12   37.25  6.909    19.33   10.95       13080    353 0.6421     0.01271  -0.001336  0.9195
   13   35.59  16.56    7.913   11.04       5942  384.4 0.6096     0.01414 -0.0009723  0.8984
   14   36.16  6.768    18.47   10.86      12210  322.9  0.587     0.01199  -0.001035  0.9852
   15    36.9  6.909     18.7   11.24      11220  302.8 0.6836     0.01124  -0.001049       1
   16   35.66  8.457    16.19   10.95      11160  368.7 0.6888     0.01279  -0.001352  0.8352
   17   52.17  26.88    13.67   11.57       2644  277.6 0.6484     0.01139 -0.0009433  0.9684
   18   37.13  6.616    18.91   11.54      11680    302 0.5828     0.01002  -0.001204   0.968
   19   38.32  7.552    19.26   11.44      11720  345.6 0.5786     0.01109  -0.001103  0.8805
   20    36.8  6.887     18.7   11.14      12580  338.3 0.6331     0.01127  -0.001162   0.977
   21   54.94  27.68    16.02   11.18       3078  332.8  0.608        0.01   -0.00117  0.9484
   22   36.52  6.822    18.73    10.9      12520  333.5 0.5455    0.008968 -0.0006723  0.8977
   23   36.66  6.274    18.86   11.46      12300  301.4 0.5469    0.008988 -0.0008627  0.9437
   24   36.49  6.538    18.76   11.12      13140  335.6 0.5413    0.008808 -0.0006479  0.8652
   25   36.46  6.055    18.99   11.35      12270  290.1 0.5469    0.009389 -0.0007963  0.9191
   26   36.86  8.054     17.2   11.56      11090  348.9 0.5142    0.007441 -0.0005945  0.9648
   27   55.36  28.48    15.82      11       2498  277.9 0.5859    0.008439 -0.0008888   0.998
   28   36.81  6.835    18.72   11.18      11430  305.1  0.582    0.007754 -0.0005806  0.9719
   29   36.75  7.601    17.94   11.14      11410  338.8 0.5474    0.007763 -0.0008119  0.9676
```

t8r8:

```
 step  step_s  gen_s  train_s  sync_s  gen_tok_s  compl    oob  ld_absmean   ld_mean  reward
    0   731.4  160.9    556.9   13.58      843.5  530.2 0.6113     0.02039 -0.001781  0.5809
    1   91.88  66.07    14.36   11.41       2372  612.3 0.6944     0.01806 -0.001669  0.5133
    2   109.6  82.65    15.35   11.53       1924  621.2 0.6447     0.01957 -0.001494     0.5
    3   66.08  41.42    13.37   11.23       3529  570.9 0.6326     0.01689 -0.001643  0.5871
    4   69.73  42.48    15.93   11.27       3328  552.3 0.6676     0.01848 -0.002206  0.5527
    5   85.78  62.39    12.01   11.32       2300  560.5 0.5988     0.01852 -0.001727  0.5711
    6   69.29  40.12    17.67   11.44       3479  545.2 0.6569     0.01707 -0.001361  0.5906
    7   64.61  41.49    11.11   11.96       3371  546.2 0.5705     0.01661  -0.00135  0.5824
    8   65.53  41.03    11.99   12.46       3805  609.8 0.6239     0.01823  -0.00173  0.5109
    9    46.6  18.21    17.41   10.93       8134  578.7 0.6314     0.01718   -0.0017  0.5719
   10   60.32  41.09    7.355   11.82       3854  618.5 0.6322     0.01683 -0.001643  0.5324
   11   47.12  17.37     18.4   11.31       8608    584 0.5981     0.01854 -0.001739  0.5477
   12   47.76     18    18.57   11.13       8575  602.8 0.6012     0.01865 -0.002059  0.5176
   13   47.06   17.3    18.78   10.92       8795  594.3 0.6572     0.01951 -0.001853  0.5074
   14   46.85  16.49    18.86   11.44       8628  555.6 0.6125     0.01819 -0.001812  0.5762
   15   46.78  16.93    18.37   11.43       8540  564.7 0.5993     0.01742 -0.001651  0.5742
   16   46.76  16.96    18.38   11.36       8783  581.8 0.6854     0.01911 -0.001516  0.4996
   17    46.1  16.38    18.38   11.27       8000  511.9 0.6969     0.01746 -0.001985  0.6008
   18    46.3  17.44    17.56   11.23       7929  540.2  0.575     0.01641 -0.001781  0.6094
   19   47.06  16.65    18.63   11.71       8627  561.2 0.5921     0.01776 -0.001357  0.5316
   20   47.17  17.58     18.2   11.33       7976  547.8 0.6286     0.01779 -0.001432  0.6137
   21   46.46  17.29    17.61   11.49       8380    566 0.6607     0.01677 -0.001602     0.6
   22   48.53  17.76    19.01    11.7       8138  564.5  0.615     0.01764  -0.00176  0.5062
   23   45.74  15.85    18.69   11.14       7840  485.3 0.6521      0.0172 -0.001896  0.5816
   24   47.63  17.82    17.76   11.99       9506  661.7  0.655     0.01857 -0.001458  0.4227
   25    46.4  16.74    18.29   11.31       8298  542.5 0.6028     0.01811 -0.001998    0.55
   26   48.68  17.86    19.03   11.72       8845    617 0.5699     0.01712 -0.001713  0.5281
   27   46.47   16.4    18.41   11.59       7968  510.4 0.5451     0.01686 -0.001495  0.6473
   28    47.9  17.56    18.45   11.84       8977  615.6 0.6561     0.01821 -0.001538  0.5293
   29    46.2  16.73    17.74   11.67       8890  580.9 0.5885     0.01754 -0.001755  0.5813
```

## Issues hit during the runs

1. **The first GSM8K launch failed for all variants at the first weight sync.**
   - Error: `XLA compilation failed` for `jit__shard_init`, in the trainer's raiden FFI buffer bind.
   - Cause: the GSM8K recipe defaults to a different Pathways server image (`raiden_20260920_v2`) than the
     DeepSWE/MLPerf recipes (`raiden_988370191`), and it does not set `RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER=0`.
   - Fix: the run script now pins both.
2. **Pod names collided across variants.** The GSM8K launcher names pods `$USER-{orch,train,roll}`. The run
   script sets `USER=$JOB_PREFIX` so the three variants can run side by side.
3. **t16r8 hung at step 0.**
   - Rollout replica 4 received a clean shutdown ~1 min into warm-up. It looks like an eviction, and JobSet
     recreated the pod.
   - The orchestrator rejected the replacement's re-registration (`duplicate worker_id`), and the first
     `train_step` never returned.
   - t16r8 was relaunched at 02:39 UTC and ran all 30 steps cleanly. Logs of the hung run are in `/tmp/fp8/logs/r8_hang/`.
   - The orchestrator refusing a restarted rollout is worth filing as a tunix bug.
4. **Leftover DeepSWE sandbox claims** (`created-by: igfp8-r8t8-orch`). The DeepSWE launcher's `stop` tries to
   delete them but swallows the RBAC error: user accounts lack delete permission on `agents.x-k8s.io` in `trellis`.

## Reproduction

Code:

- **maxtext** `igorts/fp8` @ `80391d115`: per-channel FP8 MoE, `fp8_moe_fake_quant`, `rollout_fp8_moe`, and the
  weight converter fixes.
- **tunix** `igorts/fp8-rl` @ `3afe39aae`, on top of `atwigg/mlperf` 22f5d9027:
  - `ROLLOUT_FP8` / `TRAINER_FP8` recipe switches;
  - run scripts `recipes/fp8_{gsm8k,deepswe}_35b_v5p.sh`;
  - report script `recipes/fp8_gsm8k_report.py`.

### 1. Build the image

```bash
B=/tmp/fp8/imgctx; rm -rf $B && mkdir -p $B/maxtext $B/tunix
(cd ~/git/maxtext && git archive 80391d115 | tar -x -C $B/maxtext)
(cd ~/git/tunix && git archive 3afe39aae | tar -x -C $B/tunix)
cat > $B/Dockerfile <<'EOF'
FROM gcr.io/cloud-tpu-multipod-dev/atwigg/trellis:latest
COPY maxtext /opt/src/maxtext
RUN uv pip install --no-deps --reinstall /opt/src/maxtext /opt/src/maxtext/src/maxtext/integration/vllm
# /app/tunix is an editable install. Keep the base image's generated protobuf modules (gitignored).
RUN cd /app/tunix && find . -name '*_pb2*.py' -exec cp --parents {} /tmp/ \; -print && mkdir -p /tmp/pb2 && cd /tmp && find . -maxdepth 1 -name experimental -exec mv {} /tmp/pb2/ \; && rm -rf /app/tunix
COPY tunix/tunix /app/tunix
RUN cp -rn /tmp/pb2/. /app/tunix/ && rm -rf /tmp/pb2 && python -c 'from tunix.experimental.distributed.runtime.discovery import discovery_service_pb2'
# /app/examples must match the tunix pin too.
COPY tunix/examples /app/examples
COPY tunix/pyproject.toml /app/pyproject.toml
COPY tunix/requirements /app/requirements
EOF
cd $B && docker build -t gcr.io/cloud-tpu-multipod-dev/<you>/trellis:fp8 . && docker push gcr.io/cloud-tpu-multipod-dev/<you>/trellis:fp8
```

The runs above used `gcr.io/cloud-tpu-multipod-dev/igorts/trellis:igorts-pc`, built from the same content. To use
your own image, change `TUNIX_IMAGE` in the run script.

### 2. Launch (from a tunix checkout at `3afe39aae`)

Each variant uses 96 v5p chips. The launcher switches the kubectl context to `bodaborg-v5p-nap`.

```bash
R=~/git/tunix/tunix/experimental/examples/recipes
cd /tmp   # the launcher writes scratch files to the cwd
# script variant names: bf16 = t16t16, r8 = t16r8, r8t8 = t8r8
for v in bf16 r8 r8t8; do $R/fp8_gsm8k_35b_v5p.sh $v start --dry-run > dry_$v.yaml; done   # optional: inspect
for v in bf16 r8 r8t8; do $R/fp8_gsm8k_35b_v5p.sh $v start; done
# stop (also needed after a run finishes: the trainer and rollouts keep running):
#   $R/fp8_gsm8k_35b_v5p.sh <variant> stop
```

What the variants set:

- `ROLLOUT_FP8=true` adds `"fp8_moe": true` to the vLLM `maxtext_config` and `rollout_fp8_moe=true` to the
  trainer's MaxText flags.
- `TRAINER_FP8=true` adds `fp8_moe_fake_quant=true`.
- `MAX_STEPS` (default 30) can be overridden from the environment.

The DeepSWE equivalent is `fp8_deepswe_35b_v5p.sh`, with the same variants and ~25-30 min per step.

### 3. Collect metrics

```bash
pip install wandb pandas            # any venv; needs WANDB_API_KEY for the google-trellis entity
python $R/fp8_gsm8k_report.py -v    # per-step table + means over steps >= 1
```

While a run is going, look at pods `igfp8g-<variant>-{orch,train,roll-N}` in namespace `trellis`. The trainer logs
`TRAINER_WEIGHT_SYNC_TIMING` lines, and the orchestrator logs `Weight sync finished in X seconds`.
