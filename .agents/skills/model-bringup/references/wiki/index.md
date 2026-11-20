# Model Bring-up Wiki

Reusable troubleshooting playbooks distilled from MaxText model bring-ups. Load only the page matching your active blocker to keep context lean.

## Topic Index

| Blocker / Symptom | Playbook | Summary |
| :--- | :--- | :--- |
| Compile-time or runtime TPU HBM OOM when running verification / training smoke tests (single TPU VMs or multi-host TPU clusters) | [hbm_verification.md](hbm_verification.md) | 3-stage escalation ladder ("unoptimizing" throughput for minimal chips/HBM: config/sharding flags $\to$ sub-layer remat $\to$ host-RAM expert streaming) |

## Wiki Protocol ("Read on Blocker, Merge on Fix")

1. **Read on Blocker**: Consult the Topic Index above before inventing custom workarounds when hitting OOM, compilation, or storage bottlenecks.
2. **One Page = One Atomic Topic**: Each `.md` file covers a single problem class using three sections:
   - **Symptom / When to Use**
   - **Action Ladder** (ordered from zero-code flags to code-level workarounds, with measured impact)
   - **Guardrails & Cleanup** (host DRAM/disk bounds and what must be gated or reverted before PR upload)
3. **Accumulate, Don't Duplicate**: When a bring-up uncovers a new reusable workaround, merge it into the matching page in `references/wiki/` if one exists; otherwise add a concise new page and register a row in the Topic Index above.
