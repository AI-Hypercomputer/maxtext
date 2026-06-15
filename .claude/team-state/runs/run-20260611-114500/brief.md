# Brief: Temp Compile Memory (Tmem) Deep-Dive — MaxText/JAX/XLA

## Deliverable

A standalone HTML file (`tmem-maxtext-ds-n1-se2-e256-h4096-0611.html`) suitable for:
- Converting into a blog post
- Using as slides/material for a talk/presentation
- Audience: MaxText/JAX/XLA researchers and practitioners

## Content Requirements

1. **Read and analyze in depth:**
   - `/work/jax-memory-bug-reproduce-2026-jan/ds-proxy-se2-e256-h4096.yml` — the model config
   - `/work/repo-2/maxtext` — both `upstream/main` and branch `cj/tmem-fixes-clean`
   - The full diff between the two branches, focusing on tmem-impacting changes

2. **Use `ds-proxy-se2-e256-h4096.yml` as a concrete running example** to illustrate:
   - How temp compile memory (tmem) works end-to-end: from JAX Python code → JAX tracing → XLA HLO → XLA compiler memory planning → runtime buffer allocation
   - The full call chain: `train_step` → `jax.jit` → `jax.lower` → `jax.compile` → XLA `HloModule` → buffer assignment → `memory_analysis()` → `temp_size_in_bytes`
   - What exactly "temp memory" means in XLA's buffer allocator vs. output memory vs. argument memory
   - How `remat_policy` / `checkpoint_policies.save_only_these_names` controls which activations become temp buffers vs. recomputed
   - How pipeline parallelism (`ppermute`, `shard_map`, collectives) creates temp buffers and how replacing them with local `slice`/`concat`/`pad` eliminates those buffers
   - The specific tmem reduction achieved: ~29.9 GB → ~20.4 GB, with a breakdown of each optimization's contribution

3. **Include Excalidraw-style diagrams** (embedded in the HTML) showing:
   - The compilation pipeline: Python → Jaxpr → HLO → optimized HLO → buffer assignment
   - The pipeline stage rotation: before (ppermute collectives) vs. after (local slice/concat)
   - The remat checkpoint boundary: which activations are saved vs. recomputed
   - Memory layout: argument buffers, output buffers, temp buffers in HBM

4. **Technical depth:** Do NOT simplify for a general audience. The readers are JAX/XLA practitioners who understand sharding, SPMD, HLO, and XLA passes. Go deep into:
   - XLA's `BufferAssignment` pass and how it determines temp buffer lifetimes
   - How `save_only_these_names("iteration_input", "decoder_layer_input")` maps to HLO checkpoint boundaries
   - Why `ppermute` creates larger temp buffers than `slice`+`concat` (collective vs. local ops)
   - The role of `LatencyHidingScheduler` in overlapping compute and communication

5. **No word count constraint** — be as thorough as needed.

## Source Files to Reference

- `/work/jax-memory-bug-reproduce-2026-jan/ds-proxy-se2-e256-h4096.yml`
- `/work/repo-2/maxtext/src/maxtext/layers/pipeline.py` (both upstream and branch versions)
- `/work/repo-2/maxtext/src/maxtext/layers/normalizations.py`
- `/work/repo-2/maxtext/src/maxtext/models/mixtral.py`
- `/work/repo-2/maxtext/src/maxtext/configs/types.py`
- `/work/repo-2/maxtext/src/maxtext/configs/base.yml`
- `/work/repo-2/maxtext/src/maxtext/trainers/pre_train/train.py`
- `/work/repo-2/maxtext/src/maxtext/utils/sharding.py`
- `/work/jax-memory-bug-reproduce-2026-jan/tmem-response-a-0605.md` (existing technical analysis)

## Output

Single file: `/work/jax-memory-bug-reproduce-2026-jan/tmem-maxtext-ds-n1-se2-e256-h4096-0611.html`
