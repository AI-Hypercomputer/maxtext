# Multi-Layer Numerical Parity & Error Mitigation Plan

**Date:** 2026-08-13  
**Target Architecture:** Qwen3.5 MoE (`qwen3.5-35b-a3b`), Qwen3-Next, DeepSeek-V3/V4  
**Hardware Platform:** Google Cloud TPU v5p (Shared Pathways Service / GKE)  
**Document Purpose:** Engineering roadmap and mitigation strategies to eliminate numerical divergence and prevent error accumulation across deep multi-layer transformer stacks ($32 \sim 64$ layers) between MaxText Training and vLLM Inference.

---

## 1. Problem Statement & Deep Stack Risk Analysis

In our 1-layer decoder numerical parity benchmarks on Cloud TPU v5p:
* **Float32 Layer Output (`T25_layer_output`):** $\text{Cosine Similarity} = \mathbf{1.000000}$, $\text{MAE} = \mathbf{1.73 \times 10^{-4}}$, $\text{Max Abs Error } (L_\infty) = \mathbf{7.12 \times 10^{-3}}$.
* **BFloat16 Layer Output (`T25_layer_output`):** $\text{Cosine Similarity} = \mathbf{0.999976}$, $\text{MAE} = \mathbf{1.05 \times 10^{-3}}$, $\text{Max Abs Error } (L_\infty) = \mathbf{3.12 \times 10^{-2}}$.

### The Multi-Layer Accumulation Question
While isolated MoE kernels have true machine-precision parity ($L_\infty = 2.98 \times 10^{-8}$), the attention core introduces a small summation re-association delta ($\approx 1.53 \times 10^{-5}$ in FP32) which gets multiplied by the SwiGLU MLP Lipschitz constant ($\approx 20.9\times$) to produce $7.12 \times 10^{-3}$ at Layer 1.

If unmanaged in a 32-to-64 layer model over multi-step autoregressive generation, there is a risk of:
1. **Router Misdirection:** Sensitive boundary tokens near the top-$K$ selection threshold being routed to different experts.
2. **Logit Shift:** Accumulation of small scalar biases shifting top-1 greedy token selection during generation.

---

## 2. Engineering Strategies to Eliminate & Avoid Divergence

### Strategy 1: Unified Kernel Implementation (The Gold Standard)
The most robust way to eliminate $L_\infty$ divergence is to use the **exact same attention kernel** in both training and serving:

* **Current Status:** Training uses **Tokamax Splash Attention**, while Inference uses **vLLM RPA (Pallas)**. Even with identical mathematical formulas ($e^x$), internal sequence tiling differs ($128 \times 128$ vs $256 \times 64$).
* **Action Items:**
  * **Path A (Preferred for Serving Parity):** Integrate the **Tokamax Splash Attention** backend directly into vLLM TPU inference plugins for prefill.
  * **Path B (Preferred for Training Parity):** Lower MaxText prefill attention to use **Pallas RPA** with static KV allocations during prefill evaluation runs.
* **Expected Outcome:** Eliminates the upstream seed perturbation entirely ($L_\infty = 0.000000$ at Attention Core).

---

### Strategy 2: Attention Tile & Online Softmax Alignment
If separate kernels must be maintained (e.g. dynamic paged KV memory management in vLLM vs Splash Attention in training):

* **Mechanism:** Online softmax rescales accumulators at each sequence block boundary:
  $$m_{\text{new}} = \max(m_{\text{old}}, \max(S_{\text{tile}})), \quad l_{\text{new}} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum e^{S_{\text{tile}} - m_{\text{new}}}$$
  Mismatched tile sizes ($KV_{\text{tile}} = 128$ vs $64$) create differing rescale points and summation reduction trees.
* **Action Items:**
  * Standardize `block_q = 128` and `block_kv = 128` in both Splash Attention and vLLM RPA configuration profiles.
  * Enforce consistent Flash Attention online normalizer formulation (`sa_use_base2_exp: False`, `sa_fuse_reciprocal: True`).

---

### Strategy 3: Full FP32 Attention Inner-Loop Accumulators
* **Mechanism:** Prevent intermediate truncation during attention logit scaling and value accumulation.
* **Action Items:**
  * Set `float32_logits: True` in MaxText to keep $S = \frac{Q K^T}{\sqrt{d_k}}$ in Float32 before subtracting row maximums.
  * Maintain running online softmax state ($m, l$) in FP32 registers.
  * Accumulate the probability-value dot product ($P \times V$) in FP32 before downcasting to the layer hidden state dtype.

---

### Strategy 4: Enforce High-Precision TPU MXU Dot Products
* **Mechanism:** On Cloud TPU v5p, the Matrix Multiply Unit (MXU) supports `DEFAULT`, `HIGH`, and `HIGHEST` precision dot products.
* **Action Items:**
  * Enable `matmul_precision: "highest"` (or `precision=jax.lax.Precision.HIGHEST`) for attention projections and MLP contractions in critical parity verification tests.

---

### Strategy 5: Router Gate & Expert Summation Precision Guards
* **Action Items:**
  * `float32_gate_logits: True`: Keeps router gate projections and softmax probabilities in Float32 before top-$K$ selection, preventing boundary-token misrouting.
  * `float32_weight_sum: True`: Performs the top-$K$ weighted combination ($\sum_{k=1}^K w_k \cdot \text{out}_k$) in FP32 accumulators.
  * `norm_topk_prob: True`: Normalizes expert routing probabilities uniformly across both runtimes.

---

## 3. Theoretical Bounding Mechanisms in Deep Transformers

Deep Pre-LN Transformer architectures have built-in mathematical properties that prevent errors from exploding unbounded:

```
                      ┌───────────────────────────────┐
                      │  Layer Input x_l (Bounded)    │
                      └──────────────┬────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       ┌───────────────────┐                   ┌───────────────────┐
       │   RMSNorm(x_l)    │                   │  Residual Stream  │
       │ (Resets Variance) │                   │      x_l          │
       └─────────┬─────────┘                   └─────────┬─────────┘
                 │                                       │
                 ▼                                       │
       ┌───────────────────┐                             │
       │ Sublayer f(x_l)   │                             │
       └─────────┬─────────┘                             │
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                      ┌───────────────────────────────┐
                      │ x_{l+1} = x_l + f(RMSNorm)    │
                      │ Rel Error: O(1 / sqrt(L))     │
                      └───────────────────────────────┘
```

1. **RMSNorm Variance Reset:**
   * Activations entering every sublayer are normalized by $\sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}$.
   * This resets scalar variance and prevents exponential amplitude growth ($e^{\lambda L}$) across layers.
2. **Residual Stream Attenuation ($O(1/\sqrt{L})$):**
   * In Pre-LN Transformers ($x_{l+1} = x_l + f(x_l)$), the norm of the residual stream grows as $\|x_l\| \sim O(\sqrt{L})$.
   * The relative contribution of any single layer's perturbation $\frac{\Delta f(x_l)}{\|x_l\|}$ scales as $O(1/\sqrt{L})$, dampening per-layer deviations.
3. **Directional Stability (Cosine Similarity):**
   * Cosine Similarity is **`1.000000`** in FP32 and **`0.999976`** in BF16, ensuring that the directional trajectory of hidden states remains stable.

---

## 4. Multi-Layer Verification Plan & Milestones

| Milestone | Scope | Key Objective / Deliverable | Success Criteria |
| :--- | :--- | :--- | :--- |
| **Phase 1: Depth Scaling Sweep** | 1, 2, 4, 8 Layers | Run multi-layer SPS TPU v5p benchmarks; measure $L_\infty$, MAE, and CosSim across layer depth $L$. | $\text{CosSim} \ge 0.9999$ across all 8 layers; verify error does not grow exponentially. |
| **Phase 2: Unified Attention Kernel Test** | 1 Layer & 4 Layers | Run MaxText and vLLM with identical Tokamax Splash attention backend. | $L_\infty \le 10^{-7}$ in FP32 across entire attention block. |
| **Phase 3: Top-1 Token Greedy Parity** | End-to-End Model | Execute 128-token autoregressive generation rollout comparing MaxText decode vs vLLM serving. | $100\%$ exact token-ID match across sequence rollouts. |
| **Phase 4: Automated CI Regression Guard** | Unit / E2E CI | Integrate multi-layer dump parity test into MaxText automated test suite. | Automated gate preventing numerical regressions on PRs. |

---

## 5. Summary Configuration Blueprint for Next Experiments

```yaml
# Recommended MaxText Experimental Config
attention: "flash"
use_tokamax_splash: True
sa_use_base2_exp: False        # Base-e natural exp
sa_fuse_reciprocal: True       # In-register reciprocal
float32_logits: True           # FP32 attention softmax
sparse_matmul: True            # Tokamax GMM v2
megablox: True
use_tokamax_gmm: True
use_gmm_v2: True
wi_tile_fwd_batch_seq: 256     # Aligned contraction tile
float32_gate_logits: True      # Stable routing
float32_weight_sum: True       # FP32 expert combination
norm_topk_prob: True
```
