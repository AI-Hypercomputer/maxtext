# OLMo 3.5 on Ironwood: rebase, consistency gate, measured performance loop

Tracking checklist. Tick items as they land. Full context and rationale live in the
session plan; this file is the execution record.

Reference: `allenai/OLMo-core@codex/partner-model-family-20260914`,
`src/scripts/standalone/standalone_configs.py`.

## Target and anchors

`tpu-recipes/training/ironwood/README.md`, per chip against the 2307 TFLOP/s bf16 peak:

| recipe | TF/s/chip | MFU | expert layout |
|---|---|---|---|
| llama3.1-70b (dense) | 1207 | 52.3% | dense |
| qwen3-235b-a22b | 630 | 27.3% | MoE, wide experts |
| deepseek3-671b | 608 | 26.3% | MoE, wide experts |
| gpt-oss-120b | 330 | 14.3% | MoE, many narrow experts |
| OLMoE3-3p5b (ours, measured) | ~195 | 8.45% | 512 experts, top-16 |

OLMo 3.5 is 512 experts at top-16 with expert hidden = d_model, which is the gpt-oss
class rather than the deepseek class. Goal is measured MFU above 20% on v7x, or a
defensible statement of what caps it.

MFU convention: **TF/s/device / 1153.5** (a chip holds two devices).

## Phase 0: rebase onto current main

New branch `olmo35-rebase` cut from `gagik-test-olmo`; the latter stays at `fa2db0f27`
as the fallback. Commit messages are plain one-line subjects with no attribution lines.

- [x] 0.0 This checklist written into the worktree
- [x] 0.1 Pre-rebase state recorded (HEAD `fa2db0f27`, 54 dirty paths, `/tmp/pre-rebase-status.txt`)
- [x] 0.2 Working tree backed up to `~/olmo35-prerebase-20260921.tgz` (41 MB)
- [ ] 0.3 `git switch -c olmo35-rebase`
- [ ] 0.4 Four commits: A OLMoE3 perf lane, B OLMo 3.5 shared-layer enablement, C configs + parity test, D analysis scripts and docs
- [ ] 0.5 `git fetch origin`; scope the conflict surface on the five shared files
- [ ] 0.6 `git rebase origin/main`
- [ ] 0.7 Post-rebase inspection: `olmo3-` prefix fix, the four new config fields, the four model names, tuple `num_features` in RMSNorm
- [ ] 0.8 Green-light gate: parity 8/8 + `olmoe3_test.py` + `attention_test.py`

## Phase 1: consistency gate against the reference

- [ ] 1.1 Shallow-clone OLMo-core at the partner branch
- [ ] 1.2 Confirm `partner_model.py` / `standalone_configs.py` unchanged since parity was established
- [ ] 1.3 Parity suite: 8 passed (4 param counts, 4 logits under 1e-3 relative)
- [ ] 1.4 Diff `standalone_configs.py --model-size all` geometry against the four `olmo35-*.yml` field by field
- [ ] 1.5 Record parity numbers as the gate for everything downstream

Expected parameter counts: 12,496,341,632 / 72,237,847,936 / 322,601,566,720 /
1,310,163,554,560.

## Phase 2: plumbing for real Ironwood

- [ ] 2.1 Install patched tokamax KDA from `olmo35/tokamax-kda-patched/kda` into `venv-maxtext` (0.0.13 ships `gmm_v2` but no `kda`; upstream `kda` is chunk 64 only, no bf16 switches)
- [ ] 2.2 Re-run parity after the install (KDA is on the logits path)
- [ ] 2.3 SPS smoke test, 5 steps: `Jax Backend: Pathways`, `Num_devices: 8`, clean teardown
- [ ] 2.4 Baseline: `olmo35-tiny`, seq 8192, pdb 1, 20 steps synthetic; median TF/s/device over the last 10 steps plus spread
- [ ] 2.5 No leftover `isc-proxy-agagik-*` jobs
- [ ] 2.6 Start the 4x4x4 hunt for `small` (needs >= 64 chips)

## Phase 3: lever sweep

One factor at a time against 2.4, identical steps and seed.

- [ ] 3.1 `use_gmm_v2=true` (prior: 2.06x kernel, -6.6% step)
- [ ] 3.2 `TOKAMAX_KDA_BF16_FWD=1` + `TOKAMAX_KDA_BF16_BWD=1` (grad parity 2.5e-3)
- [ ] 3.3 `TOKAMAX_KDA_DENSE_PAIRS=1`
- [ ] 3.4 `TOKAMAX_KDA_CHUNK_SIZE` 64 / 128 / 256
- [ ] 3.5 gpt-oss-120b MaxText flag set (same expert class)
- [ ] 3.6 gpt-oss-120b XLA flag set (sparse-core offload; prior 1.93x for 2-SC)
- [ ] 3.7 Cross-check qwen3-235b where it disagrees (`megablox=False`, `shard_exp_on_fsdp=False`)
- [ ] 3.8 Keep `use_random_routing` off for headline numbers

## Phase 4: per-device batch and remat

- [ ] 4.1 Sweep pdb 1 / 2 / 3 at seq 8192
- [ ] 4.2 Buy batch with `decoder_layer_input=offload`, `mlpwo=offload`
- [ ] 4.3 Re-test phase-3 winners at the best pdb
- [ ] 4.4 Record the memory wall (OOM pdb and reported temporaries)

## Phase 5: xla-shell profile loop

- [ ] 5.1 Capture xplane profile of the best config
- [ ] 5.2 `analyze_profile`: lanes, ceiling, binder, verdict
- [ ] 5.3 `roadmap --all --json`, then `--collective | --kernels | --remat | --relayout`
- [ ] 5.4 Fix in roadmap order, re-measure, stop when the floor stops moving
- [ ] 5.5 Kernel interventions only if the roadmap points at them (latent-MoE fusion prototype, `tokamax_gmm_tile_m`)
- [ ] 5.6 Record the xla-shell floor next to the measured step time

## Phase 6: perfsim cross-check and the 20% verdict

- [ ] 6.1 Re-run `scripts/olmo35_perfsim.py` at the measured operating point
- [ ] 6.2 Derive the true optimism factor (replaces the 1.7x borrowed from OLMoE3)
- [ ] 6.3 Re-issue medium and large projections with that factor
- [ ] 6.4 State the 20% verdict for `tiny` and name the binding constraint
- [ ] 6.5 If below 20%, price the architectural gap: sweep experts, `top_k`, expert width
- [ ] 6.6 If the 4x4x4 landed, repeat 2.4-5.6 for `small`

## Phase 7: decode

- [ ] 7.1 `inference_microbenchmark` for `tiny` on SPS
- [ ] 7.2 Cross-check the perfsim decode table
- [ ] 7.3 Serving flags from `tpu-recipes/inference/ironwood`
- [ ] 7.4 Hold KV-heads-divides-TP (violating it replicates the cache, up to 2x)

## Run record

One row per configuration. Filled in as runs land.

| # | phase | config / flag diff | step s | TF/s/dev | MFU | note |
|---|---|---|---|---|---|---|
| | | | | | | |
