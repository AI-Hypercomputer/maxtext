# Experiment sweep harness (single 4x8x8 slot; Kueue serializes)

Execution parallelism is 1 (cluster fits one 4x8x8 slice). Strategy:
batch-submit everything -> Kueue runs back-to-back -> harvest step times ->
fan out analysis sub-agents per finished run -> results.md.

Files:
- manifest.tsv : one row per experiment: TAG  IMAGE  FLAGS_FILE  EXTRA_MAXTEXT_ARGS  PRIORITY
- submit_sweep.sh : submits every manifest row (Kueue queues by PRIORITY)
- harvest.sh : for each finished workload, pulls steady-state step time -> results.md
- results.md : appended table (tag, change, step_s, delta_vs_16.24, profile, note)

No-build experiments (baseline image, just flags/args): C-bisect, cost_estimate, splash/gmm tiling.
Build-needed (new image): code changes (chunked variants).
