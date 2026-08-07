# Generic MaxText Remat Architecture Plan for Qwen3-80B Host Offloading

## 1. Goal & Requirements
* **Goal:** Use standard `remat_policy = "custom"` with `decoder_layer_input=offload` and `context=offload` without writing model-specific `jax.checkpoint` custom wrappers.
* **Requirement:** Offload all 48 decoder layer inputs (`decoder_layer_input`) and context tensors (`context`) to host RAM (`pinned_host`), while discarding heavy internal activations (MoE gathered tokens, GDN 1D conv projections, SwiGLU outputs) to keep TPU HBM usage `< 32 GiB`.

---

## 2. Analysis of Standard MaxText Models (Llama, DeepSeek, Gemma, Mistral)
In MaxText's core architecture (`src/maxtext/layers/nnx_decoders.py` & `src/maxtext/utils/maxtext_utils.py`):
1. `remat_policy = "custom"` resolves to `policy = jax.checkpoint_policies.save_and_offload_only_these_names(...)` using:
   * `names_which_can_be_saved = config.tensors_on_device`
   * `names_which_can_be_offloaded = config.tensors_to_offload` (e.g. `["decoder_layer_input", "context"]`)
2. In unscanned mode (`scan_layers = False`), `NNXDecoder.__call__` executes each decoder layer in a loop over `range(cfg.num_decoder_layers)` and applies `checkpointed_fn = jax.checkpoint(pure_layer_fn, policy=policy)` per layer.
3. This per-layer checkpointing:
   * Offloads `decoder_layer_input` (`[8, 2048, 3072]`) and `context` (`[8, 2048, 4, 3072]`) directly to host RAM (`pinned_host`) as 4D/5D individual layer tensors (bypassing the `jax.lax.scan` 6D stacked array layout mismatch bug).
   * Discards all un-offloaded internal layer activations (76 GB Megablox MoE tokens, SwiGLU projections, GDN 1D conv outputs) during forward pass and recomputes them one layer at a time during backward pass.

---

## 3. Generic Action Plan

| Step | Action Item | Location | Target Outcome |
|---|---|---|---|
| 1 | Enable Unscanned Offloading | `run_qwen3_80b_aot.sh` | Set `scan_layers=False` with `remat_policy=custom`, `decoder_layer_input=offload`, `context=offload` |
| 2 | Ensure Qwen3-Next Decoder Unscanned Loop Support | `src/maxtext/layers/nnx_decoders.py` | Ensure `NNXDecoder.__call__` routes Qwen3-Next through the standard per-layer `checkpointed_fn` loop when `scan_layers=False` |
| 3 | Sync to Remote TPU VM | TPU VM `t1v-n-666717f0-w-0` | Update remote workspace |
| 4 | Remote AOT Compilation Run | `run_qwen3_80b_aot.sh` in `max_venv` | Run AOT compile on TPU VM |
| 5 | Inspect HLO & Verify Host Offload | `/tmp/xla_dump/` & `/tmp/tpu_logs/` | Verify `host-send` / `host-recv` instructions for all 48 layer inputs, peak HBM `< 32 GiB`, and 100% compilation success |

---

## 4. Review & Approval Gate
This plan is submitted to the user for review.
