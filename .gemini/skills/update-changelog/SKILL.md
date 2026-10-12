---
name: update-changelog
description: >-
  Use this skill when asked to generate or update release notes in docs/release_notes.md
  with major changes, new features, architecture updates, performance improvements, bug fixes,
  and deprecations.
---

# Release Notes Updater (`docs/release_notes.md`)

This skill automates gathering, categorizing, synthesizing, and documenting major changes, bug fixes, and deprecations in MaxText to announce in upcoming releases. It can be run either on a schedule or ad-hoc for any date range.

## Workflow Steps

### 1. Gather Commits & Pull Requests (`raw_commits.txt`)

Run the helper script [gather_changes.py](../update-changelog/scripts/gather_changes.py) from the repository root with `--mode collect` to extract all valid commits since `**Last Updated**: <commit_hash>` in `docs/release_notes.md` into `raw_commits.txt`, tagged with reconciliation IDs (`[ref:#XXXX]` or `[ref:<hash>]`):

```bash
python3 .gemini/skills/update-changelog/scripts/gather_changes.py --mode collect
```

The script automatically:
- Walks `--first-parent` on `main` and resolves GitHub Merge PR numbers (`#XXXX`) to their initial feature/fix commit on the PR branch (ignoring follow-up review feedback or comment cleanup commits).
- Excludes any changes in `tests/`, `.github/`, `.gemini/`, `src/maxtext/training_engine/`, `src/maxtext/experimental/` (`experimental/`), `src/dependencies/dockerfiles/`, `src/dependencies/scripts/`, `tools/`, `pytest.ini`, `README.md`, `*Dockerfile`, and `docs/release_notes.md`. Keep this list in sync with `EXCLUDED_PATH_PREFIXES` / `EXCLUDED_FILES` in the script.
- Excludes container/Docker image vulnerability fixes, Docker registry migrations (e.g., Artifact Registry), Gemini/agent skill updates, `pytest-split` / test runner splitting, `Latest News` announcements, `"No public description"` commits, broken link / `toctree` / Sphinx doc-build fixes, and test-only fixes.
- Strips Copybara/Piper internal metadata (`PiperOrigin-RevId`).
- Reads `**Last Updated**: <commit_hash>` from `docs/release_notes.md` (if present) so subsequent runs automatically inspect only commits in `<last_commit>..HEAD`.
- Consolidates multi-part PR series (e.g., `[NNX] Delete Linen (1/5)..(5/5)`, `FP8`, `DeepSeek-V4`) into single unified entries.
- Outputs a flat list of cleaned candidate commits with ReadTheDocs links and a tracking tag `[ref:...]` on every item in `raw_commits.txt` to guarantee no commit is dropped during synthesis.

### 2. Categorize & Polish into `new_notes.md` (Without Dropping Any `[ref:...]` IDs)

Read `raw_commits.txt` and categorize + rewrite the entries into a polished, synthesized Markdown block saved to `new_notes.md`.

> **CRITICAL — Do Not Drop Any `[ref:...]` IDs**:
> - Every `[ref:...]` ID present in `raw_commits.txt` **must** appear in `new_notes.md`.
> - When combining multiple related commits/PRs into a single cohesive bullet, include all of their IDs on that bullet (e.g., `[ref:#5051, #5052, #5053]`).
> - Step 3 runs a deterministic reconciliation check that compares all `[ref:...]` IDs in `raw_commits.txt` against `new_notes.md` and fails if any ID is missing, before automatically stripping the temporary `[ref:...]` tags upon injection.

**Semantic Categorization by Gemini**:
- Inspect each entry's summary in `raw_commits.txt` (and run `git show <hash>` if needed) to place it into the appropriate section (`#### Changes`, `#### Bug Fixes`, or `#### Deprecations`) and category.
- **Disambiguating Model Changes (`DeepSeek`, `Qwen`, `Gemma`, `Llama`, `Mixtral`, `GPT`, etc.)**: Model commits often impact more than just pre-training. Categorize them based on their actual scope:
  - Full model onboarding, decoders, layers, or cross-cutting architecture support -> **Models**
  - Vision, audio, speech encoders/processors (e.g., Qwen3-VL, Gemma3 vision, SigLIP/CLIP) -> **Multimodal**
  - RL (GRPO/DPO), SFT, LoRA/QLoRA, vLLM rollouts, or Tunix integration for a model -> **Post-Training**
  - Sharding (FSDP/FSDP2/SPMD/Tensor Parallel/Megatron), quantization (FP8/FP4/Qwix/AQT), or kernel speedups -> **Performance**
  - Attention mechanisms (Splash/Ring/CP/Ulysses), RoPE/MRoPE, MTP, Block Diffusion, or DiLoCo -> **Pre-Training**
  - If a major PR or PR series spans multiple areas (e.g., adding a new model along with its multimodal processor or RL recipe), either group the architectural onboarding under **Models** and mention the specific capability under **Multimodal** / **Post-Training**, or consolidate under the primary user-facing category.

1. **`#### Changes`**:
   - **Group related PRs**: If multiple commits contribute to a single feature (e.g., FP8 Orbax restoration and scale tensor ingestion), combine them into a single cohesive bullet and merge their `[ref:...]` tags.
   - **Links**: Do **not** append GitHub PR links (`[PR #XXXX](...)`) to bullets. Only include documentation links from ReadTheDocs (`https://maxtext.readthedocs.io/en/latest/...`) when relevant documentation, guides, or tutorials exist for the change.
   - **Use MaxText Category Headings**:
     - **Models** (new model integrations, decoders, layers, MoE routing, Flax NNX migration)
     - **Multimodal** (vision/audio/speech encoders, multimodal processors, CLIP, SigLIP, VL models)
     - **Pre-Training** (attention kernels, Context Parallelism / Ring / Ulysses, RoPE/MRoPE, MTP, Block Diffusion, DiLoCo)
     - **Post-Training** (GRPO, DPO, SFT, LoRA/QLoRA, Raiden weight sync, vLLM rollout)
     - **Performance** (sharding, SPMD, FSDP/FSDP2, tensor parallelism, Megatron, FP8/FP4, Qwix, AQT, Muon optimizer, SparseCore, GEMM overlap)
     - **Checkpointing / Goodput** (Orbax, Zarr3, OCDBT, multi-tier checkpointing, elasticity)
     - **Usability** (evaluation harnesses, tutorials, guides, documentation, configs)
2. **`#### Bug Fixes`**:
   - Merge notable user-facing bug fixes into a single flat list under `#### Bug Fixes` (without splitting into category subheadings or adding PR links), keeping their `[ref:...]` tags.
3. **`#### Deprecations`**:
   - Merge all deprecations into a single unified list under `#### Deprecations` (without splitting into category subheadings), keeping their `[ref:...]` tags. Document deprecated or removed configuration flags, legacy modules/layers (e.g., Flax Linen removal, `pure_nnx` config flags), deprecated CLI scripts, or dropped dependencies.

### 3. Reconciliation Check (`--mode verify`) & Deterministic Injection (`--mode inject`)

1. **Reconciliation Check (`--mode verify`)**:
   Verify that `new_notes.md` contains every `[ref:...]` ID from `raw_commits.txt`:

   ```bash
   python3 .gemini/skills/update-changelog/scripts/gather_changes.py --mode verify
   ```

   If any commit/PR ID was dropped during synthesis, the script exits with code `1` and prints the exact dropped IDs. Fix `new_notes.md` so all IDs are accounted for and re-run `--mode verify`.

2. **Deterministic Injection (`--mode inject`)**:
   Once verification passes, strip all temporary `[ref:...]` tags from `new_notes.md`, update `**Last Updated**: <head_commit_hash>`, and deterministically splice the clean Markdown block into [docs/release_notes.md](../../../docs/release_notes.md):

   ```bash
   python3 .gemini/skills/update-changelog/scripts/gather_changes.py --mode inject
   ```

   Keep `raw_commits.txt` and `new_notes.md` in the workspace so the CI workflow can also run `--mode verify` and `--mode inject` before committing `docs/release_notes.md`.

### 4. Validation

Verify your changes before completing the task:
1. Run `git diff docs/release_notes.md` to ensure clean Markdown formatting and that `**Last Updated**: <head_commit_hash>` is updated.
2. Confirm that bullets do not contain GitHub PR links or leftover `[ref:...]` tags, and that any documentation links point to ReadTheDocs (`https://maxtext.readthedocs.io/en/latest/...`).
3. Re-run `python3 .gemini/skills/update-changelog/scripts/gather_changes.py --mode collect` to verify that it detects `**Last Updated**: <head_commit_hash>` and reports `No new commits`.
