# Mini-Model-Verification

Pick a representative trimmed version of the target model and verify parity against the HuggingFace PyTorch reference before scaling to full checkpoints.

## Choosing a Trimmed Model
Choose a representative mini model—for example, keeping all other configurations identical to the original model while selecting 2 layer cycles (plus any unrolled prefix/remainder layers when `num_layers % cycle != 0`), running in `fp32` (`matmul_precision=highest`, `dropout_rate=0.0`) with weights mapped via `to_maxtext`. If testing from scratch initialization rather than sliced real weights, randomize zero- or identity-initialized parameters (centered norms, residual gates, router biases) so branches and gradients do not trivially collapse.

## Verification Checklist (Apply Where Applicable)

1. **Design challenging inputs deliberately**
   - *Sequence, window & layout boundaries*: Test `seq_len = 1`, non-multiples of attention tile/chunk sizes, long RoPE prefixes, and `seq_len > sliding_window_size` (so tokens cross local windows while global layers still attend to them). If `num_layers` is not divisible by `inhomogeneous_layer_cycle_interval`, choose a layer count that exercises both unrolled remainder layers and scanned cycle blocks.
   - *Heterogeneous batches & padding sides*: Combine short and full-length rows (`inputs_segmentation == 0` on right and left padding); verify mutating padded token IDs leaves valid outputs/losses invariant and batched rows match single-row execution (`mean_batch_diff <= 1e-4`). For multimodal models (`use_multimodal=True`), test both mixed-modality batches and text-only batches (`images=None`).
   - *Packed-document boundaries*: Pack multiple segments (`inputs_segmentation` = `1, 2, ...` with `inputs_position` reset to `0` per segment and `targets_segmentation` masking prompt prefixes, doc boundaries, and padding); verify exact zero cross-document attention/loss leakage and position-invariance on NoPE layers vs position-sensitivity on RoPE layers.
   - *Chat templates & special tokens*: Test `tokenizer.apply_chat_template` variants (multi-turn, system, tool-use, thinking tags), BOS/EOS/PAD/multimodal fusion tokens, and boundary IDs (`0`, `vocab_size - 1`, and PLE vocab bounds).

2. **Check forward outputs, masks, and KV-cache equivalence at meaningful boundaries**
   - *Explicit mask parity*: Assert 0 mismatches on causal, sliding-window, packed-segment, and multimodal bidirectional masks before comparing activations.
   - *Component & full-model parity*: Compare teacher-forced sublayer outputs (embeddings, centered/scaleless/gated RMSNorms, per-head QK-norm & RoPE/NoPE, attention output gates, KV-shared donor/recipient layers, MLP/MoE, temperature-scaled residual gates, final norm) and complete mini-model logits across `scan_layers=True` and `scan_layers=False`. When `fp32` summation order differs near tolerance limits, compare both against an HF `float64` ground-truth run.
   - *KV-cache prefill + autoregressive step parity*: Verify `MODEL_MODE_PREFILL` + step-by-step `MODEL_MODE_AUTOREGRESSIVE` (`scan_layers=True` and `False`, including cache wraparound past `sliding_window_size`) matches full causal `MODEL_MODE_TRAIN` forward logits.

3. **Give MoE its own coverage requirements** *(if `num_experts > 1`)*
   - *Router decision parity*: Check `fp32` router logits, score function (`softmax` vs `sigmoid`), selection-only correction bias (`e_score_correction_bias` / `MoEBiasVar`) vs weight normalization (`norm_topk_prob`), `routed_scaling_factor`, and top-$k$ index agreement on non-tie tokens.
   - *Deliberate routing patterns*: Cover all experts, all tokens concentrated on the same experts, starved/empty experts (`ragged_dot` tail masking), the last expert (`num_experts - 1`), width bottlenecks (`moe_expert_input_dim`), per-expert post-norms, shared experts, padding exclusion, and layer-order extraction of routed experts across scanned vs unscanned blocks.
   - *Forced routing*: Inject reference indices via `forced_routed_experts` (`[batch, seq, top_k]` or `[batch, seq, num_layers, top_k]`, `-1` on padding) to isolate expert/weight parity from boundary tie-flips, and verify two distinct forced routings and unforced routing yield distinct losses.

4. **Verify every enabled head and the actual training objective** *(if applicable)*
   - *All enabled heads*: Check `lm_head` before and after output multipliers and `tanh` soft-capping (verifying tied `lm_head` does not double-apply `embedding_multiplier`), vocab-tiling (`num_vocab_tiling > 1`) if used, and enabled secondary heads (`mtp_block`, `indexer`, block-diffusion).
   - *Decomposed & combined losses*: Inspect `train.loss_fn` `aux` terms before aggregation (`xent_sum`, `total_weights`, `z_loss`, `moe_lb_loss`, `mtp_loss`, `indexer_loss`); align pre-shifted `(inputs, targets)` with HF `model(labels=...)`, and test pure CE, SFT completion-only loss (`targets_segmentation`), and auxiliary terms separately and together (including under `gradient_accumulation_steps > 1`).

5. **Make backward checks sensitive and independent** *(if training applies)*
   - *Per-parameter & input gradients*: Compare `fp32` gradients for every `nnx.Param` (including individual routed experts, `scan_layers=True/False`) and wrt input embeddings (`dx`) using both relative L2 tensor error (`||g_jax - g_ref|| / ||g_ref||`, catching missing collective scaling) and scale-relative elementwise tolerances, asserting every trainable leaf is finite and non-zero.
   - *Block-level VJP probes & mutation sensitivity*: Probe each custom block in isolation with an asymmetric random upstream cotangent `dy` (`jnp.sum(block_out * dy)` vs `pt_out.backward(pt_dy)`). Perturb custom scalars/multipliers, gates, soft-caps, and router weights (`PerturbSpec` / mutation checks) and assert parity checks fail so no test is vacuous.

6. **Test actual multi-step updates and state propagation** *(if training applies)*
   - *Multi-step optimizer parity*: Run 3–5+ real `train.train_step` updates (e.g., AdamW with LR schedule and gradient clipping) using changing batches; compare step-by-step losses, final parameters, and optimizer moments (`mu` / `nu`) against PyTorch.
   - *Non-`Param` state & checkpoint continuity*: Verify out-of-optimizer state updates (`routed_bias` / `MoEBiasVar`, `_overwrite_with_gradient`), clean `nnx.Intermediate` popping (unchanged state treedef across steps, zero JIT re-traces), and mid-run Orbax save $\to$ restore producing identical next-step loss.
   - *Trainable / frozen selection*: Confirm `nnx.Param` maps 1:1 onto HF `requires_grad` parameters, and assert frozen leaves (`trainable_parameters_mask`, frozen vision/audio encoders, non-trainable routing biases, or base weights under LoRA) allocate no unexpected optimizer state and remain bitwise unchanged.

7. **Verify checkpoint round-trip, `remat`, and multi-device sharding parity**
   - *Checkpoint bijection & dtypes*: Verify `to_maxtext` $\leftrightarrow$ `to_huggingface` round-trips losslessly (`scanned` & `unscanned`, zero missing/unmapped keys) and preserves strict `float32` module dtypes (e.g., `_keep_in_fp32_modules_strict` routers and residual gates) plus non-trainable state (`MoEBiasVar`) under `bfloat16` saves.
   - *Sharded mesh & `remat` equivalence*: Run forward, loss, and backward checks on a multi-device mesh (`fsdp > 1`, `tensor > 1`, and `expert > 1` if supported—or verify explicit guards on unsupported parallelism) and across `remat_policy` settings (`none` vs `full`), asserting parity with single-device unrematerialized `fp32` outputs and gradients.
