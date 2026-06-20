# Cross-layer prefetch — feasibility findings (fork investigation, 2026-06-20)

No code changes made (branch left clean) — investigation determined neither clean path is a
small, one-shot-validatable change; a half-built unvalidated refactor would be worse than this.

## Param structure (confirmed)
Stacked moe params live at `self.variables["params"]["moe_layers"]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_0"]`
(and `wi_1`, `wo`), shape `[L, E, embed, mlp]`. Accessible ONLY at apply (`not is_mutable_collection
("params")`), not at init. Precedent: `deepseek_batchsplit_fp8.fetch_weights(params,...)` indexes
exactly these keys.

## Two CLEAN paths (and why the obvious hybrid is broken)
- **Path A — lift to parent (the directive's make-or-break).** RoutedMoE must STOP creating
  `wi_0/wi_1` (it's a bridged nnx module that auto-manages them) and the Decoder must own a stacked
  `[L,...]` param, threaded into the scanned layer as `xs`. Reuses the existing `nn.scan` + layer
  forward. **Risk = the nnx↔linen bridge:** removing an `nnx.Param` from a `to_linen`-bridged module
  and re-owning it in the linen parent is genuinely uncertain (this is the exact `nnx.scan` fragility
  the memory flags). This needs a designed bridge change, not a one-shot guess.
- **Path B — apply-time extract + custom scan (PROVEN).** Exactly what batchsplit does: at apply,
  extract `self.variables["params"]["moe_layers"][...]`, build the shifted slice, run a `jax.lax.scan`
  with carry=(activation, gathered_w01). Gradient-correct (flax tracks the `self.variables` read; no
  `variable_axes` double-management). **Cost = re-implementing the layer forward (attention+MoE)
  manually** like `scan_batch_split_layers` (~the batchsplit module). Big, and structurally it's "the
  batchsplit machinery" the user wanted to avoid — though used for prefetch, not batch-split.
- **Hybrid (keep `nn.scan` + ALSO pass extracted params as broadcast `xs`) — DO NOT.** The params
  would be both `variable_axes`-managed AND extracted-data → **gradient double-management.** This is
  the trap; both clean paths exist precisely to avoid it (A: params only parent; B: params only data).

## Recommendation
Do **Path A**, but the make-or-break must be a *designed* step, AOT-gated:
1. In `RoutedMoE.__init__`, gate `wi_0/wi_1` creation off under `moe_xlayer_prefetch`; make
   `sparse_matmul`/`gather_routed_weights` take them as args (extend the existing `pregathered_weights`
   seam — already the injection point).
2. In `decoders.py` non-pipeline DeepSeek path, create the parent stacked `W01_sharded` (linen
   `self.param`, sharding = layer-axis prepended to `wi_kernel_axes`), match per-layer `kernel_init`
   RNG folding for sane init.
3. Pass `W01_sharded` slice as a scanned `xs` (in_axes=0, NOT broadcast) + carry the gathered w01.
4. AOT after EACH — step 1's AOT (params-as-arg, no lift yet, pass the layer's own sliced param back
   in) is the cheapest probe of whether the seam works before touching the parent/scan.

If Path A's bridge step fails AOT, Path B is the proven fallback (accept the forward re-implementation).
The forward-only scope + `wo`-stays-in-layer + pad-the-xs decisions all still hold.
