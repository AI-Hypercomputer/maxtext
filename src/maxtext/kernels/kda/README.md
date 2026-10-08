# KDA kernels (`maxtext/kernels/kda`)

Pallas TPU kernels for **Kimi Delta Attention** (KDA): the chunked gated delta
rule with a per-key-channel decay, forward and backward, as used by the KDA
layers of OLMoE3 / OLMo 3.5 (`models/olmoe3.py`, `kda_pallas_fused=true`).

```python
from maxtext.src.maxtext.kernels import kda

out = kda.chunk_kda(q, k, v, log_decay, beta, segment_ids=seg,
                    chunk_size=128, state_dtype=jnp.float32,
                    layout="token_major", outer_block=32,
                    inverse_precision=jax.lax.Precision.HIGH)
```

`chunk_kda` is differentiable (custom VJP) and runs per device: under SPMD,
call it inside `shard_map` with the batch (or head) axes split and the sequence
whole (`OLMoE3KimiDeltaAttention._chunk_kda` does this).

## Math

Per head, with state `S_t` (`dk x dv`), `alpha_t = exp(log_decay_t)` (per key
channel) and `beta_t` (scalar):

```
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
o_t = S_t^T q_t
```

i.e. forget, predict `v_hat = (Diag(alpha_t) S_{t-1})^T k_t`, write the residual
`beta_t (v_t - v_hat)` and read. The state is zeroed at every document start
(`segment_ids` changes). Because `Diag(alpha)` and `(I - beta k k^T)` do not
commute, the decay cannot be pulled out of the chunk algebra as a scalar mask.

Chunked form (`C` tokens per chunk, `g` the in-chunk cumulative log-decay,
`g_i - g_j <= 0` for `j <= i`):

```
kk_ij     = sum_d k_id k_jd exp(g_id - g_jd)   (j < i, same document)
scores_ij = sum_d q_id k_jd exp(g_id - g_jd)   (j <= i, same document)
X         = (I + Diag(beta) kk)^-1              (unit lower triangular)
delta     = X (beta v - beta (k e^g) S_c)       = u - w S_c      (WY form)
out       = (q e^g) S_c + scores delta
S_{c+1}   = Diag(e^{g_last}) S_c + (k e^{g_last - g})^T delta
```

Only `S_c` is sequential; everything else is per chunk.

* **Intra-chunk matrices** (`intra.py`): rows go in sub-blocks of 8 (and outer
  blocks of `outer_block`); against earlier columns a row's decay factors
  through its block's first row `p`: `exp(g_i - g_j) = exp(g_i - g_p) exp(g_p -
  g_j)`, both exponents `<= 0`, so below-diagonal blocks are overflow-free MXU
  matmuls. Only diagonal sub-blocks use the exact per-pair factor (masked before
  the `exp`). The naive factorization `(q e^g)(k e^-g)^T` overflows f32 once the
  cumulative decay passes ~88 nats, which real KDA decays reach.
* **Inverse**: exact block doubling from 1x1 blocks,
  `X_2s = X_s - X_s L_s X_s` (`fused._inverse`), f32 matmuls at
  `inverse_precision`. Its VJP is `dM = -tril(X^T dX X^T, -1)`.
* **Backward**: walks the chunks in reverse carrying `R = dL/dS_{c+1}` in VMEM;
  the intra-chunk VJP uses the closed form `dg = q*dQ + k*(dK_row - dK_col)`
  plus the pivot-row correction (keeps `sum dg = 0` per chunk exactly, so the
  log-decay gradient, a reverse cumsum, does not drift).

## Layouts

| `layout`        | kernel HBM layout                         | cumulative decay |
| --------------- | ----------------------------------------- | ---------------- |
| `token_major`   | reads/writes `[B, T, H * d]` directly      | in-kernel (3-pass bf16 tril matmul, exact to f32) |
| `chunk_major`   | `[B * H, T / C, C, d]` (XLA transposes)    | XLA (`HIGHEST` tril matmul) |

Inputs and outputs of `chunk_kda` are always `[B, T, H, d]`. `token_major`
needs `dk`, `dv` multiples of 128 on TPU (unless `H == 1`). `T` is zero-padded
to a multiple of `chunk_size`. The kernel grid is `(B * H, T / C)` with the
first axis `parallel` (both v4 megacore TensorCores) and the chunk axis
`arbitrary` (sequential, state in VMEM).

## Numerics

* All decay algebra (`g`, `exp`, masks), the state `S` (with
  `state_dtype=float32`), the inverse and all accumulations are f32.
* Matmul operands are rounded to bf16 (what default-precision f32 matmuls do on
  TPU), except the triangular inverse and its VJP (`inverse_precision`;
  `HIGH` = 3-pass bf16 ~ 6e-6 relative error, `HIGHEST` = 6-pass) and the
  cumulative sum (exact 3-way bf16 split).
* `state_dtype=bfloat16` additionally rounds `S` to bf16 at every chunk boundary
  and casts q/k/v to bf16 (half the HBM traffic; a quality trade-off).
* `output_dtype` only sets the store dtype of the f32-accumulated output (bf16
  output == f32 output cast to bf16, bit for bit, and the bf16 cotangent is
  read at half the bytes).
* vs. the f32 per-token recurrence (`kda_recurrent_reference`): ~1e-3 relative
  (outputs) at f32 state; tests in `tests/unit/olmoe3_test.py`
  (`OLMoE3FusedKDAKernelTest`).

## Residuals and remat

The VJP forward saves, per batch x head and chunk: `S_c` (`dk x dv`, f32 or
bf16), `delta` (`C x dv` bf16), `X` and `kk` packed into one `C x C` f32 matrix
(kk strictly lower, `(X - I)^T` strictly upper; exact) and `scores` (`C x C`
bf16). Under activation remat these are normally dropped and the
forward kernel runs again in the backward.

* `save_residuals=True` tags the output and the residuals with
  `checkpoint_name("kda_fused")`; a policy that saves that name (MaxText:
  `remat_policy=custom kda_fused=device`) keeps them, and the recompute pass no
  longer contains the forward kernel.
* `state_stride` picks which `S_c` are residuals: `1` all; `2` even chunks only
  (the backward kernel rebuilds an odd chunk's state from the previous chunk's
  saved state, k, decay and delta, one extra `[dk, C] x [C, dv]` matmul); `0`
  none (`fused._states_call`, a separate light pass, rebuilds them all from the
  deltas). Same arithmetic as the forward's recurrence, so all are bit-identical
  (checked against the pre-package kernel on TPU v4).

HBM per call (`kda.residual_bytes_per_layer`), olmoe3-3p5b per chip
(`B=1, T=8192, H=8, dk=256, dv=512, C=128`):

| tensor | MiB |
| ------ | --- |
| output (bf16) | 64 |
| delta (bf16) | 64 |
| X and kk, packed (f32) | 32 |
| scores (bf16) | 16 |
| **total, `state_stride=0`** | **176** |
| states, `state_stride=2` (f32) | 128 |
| **total, `state_stride=2`** | **304** |
| states, `state_stride=1` (f32) | 256 |
| **total, `state_stride=1`** | **432** |

## Files

* `api.py`: `chunk_kda`, `segment_resets`, `residual_bytes_per_layer`.
* `fused.py`: the fused forward / states / backward kernels and their custom VJPs
  (`kda_fused`, `kda_fused_token_major`).
* `intra.py`: intra-chunk matrix kernels (forward rows, backward), also usable alone.
* `scan.py`: chunk-state recurrence kernels used by the unfused sub-block path
  (`kda_pallas_scan`, `kda_pallas_chunk`).
* `reference.py`: `kda_recurrent_reference`, the per-token recurrence in JAX.

The old modules `kernels/kda_fused.py`, `kda_intra.py`, `kda_scan.py` re-export
these for existing imports.
