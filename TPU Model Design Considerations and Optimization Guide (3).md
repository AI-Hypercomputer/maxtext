# **TPU Model Design Considerations and Optimization Guide**

| \#begin-approvals-addon-section Username Role Status Last change [andig](http://teams/andig) Approver 🟢 Approved Jul 24, 2026 [carlosaraya](http://teams/carlosaraya) Approver 🟢 Approved Jul 24, 2026 [hisipra](http://teams/hisipra) Approver 🟢 Approved Aug 25, 2026 [liquncheng](http://teams/liquncheng) Approver 🟡 Pending Jul 24, 2026 [rjx](http://teams/rjx) Approver 🟢 Approved Jul 24, 2026 [sivaibhav](http://teams/sivaibhav) Approver 🟡 Pending Jul 24, 2026 [vipannalla](http://teams/vipannalla) Approver 🟡 Pending Jul 31, 2026 [deepakspatil](http://teams/deepakspatil) Reviewer 🟢 LGTM Jul 24, 2026 [mattdavidow](http://teams/mattdavidow) Reviewer 🟢 LGTM Aug 31, 2026 [ranran](http://teams/ranran) Reviewer 🟢 LGTM Jul 27, 2026      ![][image1] For more information, see [go/g3a-approvals-reviewing](https://goto.google.com/g3a-approvals-reviewing)&nbsp; |
| ----- |

&nbsp;

**Authors**: [Gagik Amirkhanyan](mailto:agagik@google.com) [Vaibhav Singh](mailto:sivaibhav@google.com)

### **Purpose**&nbsp;

This document is a practical guide to designing and running models efficiently on TPUs, covering both training and serving. Many of the rules improve efficiency in general, so the benefits often carry over to other accelerators. The guide applies to LLMs broadly. MoE models get the deepest coverage (Principles 3-5, 9\) because they benefit the most from having TPU-specific design considerations.

### **Designing models for Ironwood (TPU v7x) and future TPU generations — training and inference**

Performance tuning starts with model definition. Architectural choices made before the first training step set permanent ceilings on MFU (Model FLOP Utilization), cost, and serving latency. This guide gives concrete, hardware-grounded design rules for teams building or adapting models for TPUs, organized into three groups:

&nbsp;

1. **Shape rules**: pure arithmetic, near-zero model-quality risk. Adopt always.  
2. **Architecture choices**: involve quality tradeoffs. Validate with ablations.  
3. **Systems mapping**: no model change required. Pure engineering.

&nbsp;

**Where to start.** The shape rules apply to anyone designing a new model; use the table below to filter the rest, with the Targets column routing training vs serving owners and the Scope column marking the few MoE-only rules. If you are adapting an existing model rather than designing one, the systems-mapping section applies without touching the checkpoint. The general rules hold for any large-matmul architecture (dense LLMs, multimodal and world models), and the doc is structured to grow model- and workload-specific sections over time.

## **Principles at a glance**

| \# | Group | Guideline Summary | Scope | Targets (Inference/Training) | Quality risk |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | Shape | Every model reduction dimension (or output dimension) is a multiple of 256 (most-missed: head\_dim \= 256), counts like num heads/batch are exempt | general | both | \~none |
| 2 | Shape | Larger scale up domain for TPU (FSDP/TP/EP) coupled with multislice (FSDP/DP) helps strong scaling to large compute; pipelining is feasible as well. (details). (GBS/sparsity/DCN-AI replicas; use pipelining beyond) | general | training | \~none |
| 3 | Shape | Static shapes everywhere; where the model needs raggedness (MoE routing), TPUs support it via mechanism like scalar prefetch (e.g. Megablox, GMM v2 kernels), which is what makes dropless routing efficient | general \+ MoE | both | \~none |
| 4 | Arch | Compress token communication volumes;  e.g. LatentMoE  reinvest the saving in expert capacity (e.g.); low precision a2a (fp8 or lower, also seen in 2026 frontier OSS models) | MoE | both | Ablate (latentMoE), low-precision a2a is quality neutral |
| 5 | Arch | If quality-neutral, higher compute intensity is preferred (e.g. shared experts, larger d\_ff) | MoE | both | ablate |
| 6 | Arch | Low-precision as architecture: bound dynamic ranges (QK-norm, z-loss) so cheap static or coarse scaling holds quality. Sub-channel (block-scaled) quantization: TPU v8 and onwards supports block scaling in the MXU; v7x and earlier run block scaling on the VPU and are therefore sensitive to the MXU:VPU ratio (240:1, Principle 9). Recommended block size 512 or above on v7x; v8 has no such constraint, for MXFP8 or MXFP4 | general | both | ablate |
| 7 | Arch | Hybrid attention: Sub-quadratic attention, or local/global interleave default, SSM aligns with compute ratio trends for TPUs (Frontier models already exhibit this trend; more KV friendly for decode and compute friendly for prefill at large context)&nbsp; | general | both | ablate |
| 8.1 | Arch | Decoding is conventionally memory bandwidth bound; However leveraging large scaleout low network diameter topologies e.g. boardfly help perform beyond bw ratios in latency bound regimes.&nbsp; | general | Inference | none |
| 8.2 | Arch | An approach such as MTP speculative decoding, can leverage the TPU/GPU compute resource more effectively: several tokens verified per weight read. Choose the drafter at design time. | general | Inference | low |
| 9 | Arch | For effective VPU/MXU overlap, size the VPU vs MXU op with hw ratio awareness, e.g. for v7x fp8MXU-f32-VPU ratio is 240:1, this directly informs the minimum blocks size (details)&nbsp; | general \+ MoE | training | \~none |
| 10 | Systems | Map every parallel axis to one or more  physical link (D2D → ICI → DCN), (sub-axis is a special case, possible but needs careful tradeoff considerations) | general \+ MoE | training | none |
| 11 | Systems | Use Pallas for fine-grained Memory/Compute pipelining control. Published kernels for new configs will require block size tuning.&nbsp; | general | both | none |
| 12 | Systems | For effective latency/throughput trade-off often prefill/decode will require different sharding, batchsize and slice shapes. Disaggregation is recommended to maximize performance. | general | general | low |
| 13 | Systems | For bf16 dominant compute workload (e.g. quadratic attention with large sequence lengths, where fp8 is not a direct option, we recommend using multi-pass bf16 algorithm (will be available on v8\* onwards). It’s especially useful because of higher fp8:bf16 compute ratios in v8\* generation | general | general | low |

&nbsp;

> **Measured evidence.** Alongside the published sources, notes marked **Measured** carry results from our own Ironwood profiling program: open 20–30B MoEs (a Qwen3-30B-derived LatentMoE variant; GPT-OSS-20B) plus dense controls, single slice (except for the two-slice comparison in Principle 2), bf16, training shapes up to 32 K sequence — all against tuned-systems baselines. Treat the trends and rankings as transferable, the exact percentages as config-specific.

&nbsp;

&nbsp;

## **Hardware facts that drive performance (Ironwood / TPU v7x)**

| Property | Value | Design consequence |
| :---- | :---- | :---- |
| MXU (matrix multiply unit) | 256×256 systolic array | Make every contracting dim a multiple of 256: matmuls run in 256-wide tiles, and remainders are zero-padded, wasting cycles |
| Peak FP8 | 4,614 TFLOPS/chip (2× BF16) | Default to FP8 for training and serving: 8-bit operands double math throughput and halve bytes moved. |
| HBM | 192 GB @ 7.38 TB/s | An op needs ≈ 600 FLOPs per byte it touches (FP8; ≈ 310 BF16) from HBM to stay compute-bound instead of waiting on HBM. Elementwise ops sit far below that line (minimize and fuse them), and decode re-reads the weights every token, so it needs about that many concurrent tokens per chip, or fewer bytes per token (e.g. lower-precision weights) |
| VMEM | 64 MB per TensorCore | Kernels work tile by tile: move a tile from HBM into VMEM, compute on it, write it back, with the next tile's load pipelined behind the compute; larger tiles mean fewer HBM↔VMEM transfers, but all live tiles must fit in 64 MB, which bounds kernel block shapes (including flash-attention blocks) |
| 1D arithmetic intensity over ICI (inter-chip interconnect) | \~11,500 FLOPs/byte (BF16; \~23,000 FP8; 200 GB/s ICI) | A collective stays hidden only if compute-per-byte transferred over ICI exceeds this ratio; TP moves activations every layer and rarely clears it, so avoid TP over ICI in training |
| Chip layout | Two chiplets per chip, each with a TensorCore and its own 96 GB HBM; die-to-die (D2D) link \~6× faster than a 1D ICI link | Map TP=2 onto the two chiplets so tensor-parallel traffic stays on the fast D2D link inside the chip; any wider TP must cross the slower ICI. Memory is per chiplet: each TensorCore addresses its own 96 GB, not a shared 192 GB |
| Topology | 3D torus (ICI) \+ DCN (data-center network) across pods | EP/CP need torus bandwidth every layer; DP and PP tolerate DCN best (one fixed-size sync per step or stage), FSDP over DCN re-gathers weights every accumulation microbatch, so reserve it for models too large for one slice's HBM. |
| SparseCore | Offload engines for non-matmul work (irregular memory access, collectives) | Embedding lookups, MoE dispatch (sorts, gathers, scatters), and collectives can run off the TensorCore, overlapped with compute |

Parallelism shorthand used throughout: DP \= data, TP \= tensor, EP \= expert, CP \= context, PP \= pipeline parallelism; FSDP \= fully-sharded data parallel.

## **Shape rules (adopt always)**

### **1\. Every model reduction dimension (or output dimension) is a multiple of 256 (d\_model, FFN dim, head\_dim; not counts like num heads or layers)**

Make d\_model, FFN/expert intermediate dims, latent dims, attention projections, and vocab padding all multiples of 256, matching the MXU's 256×256 array. Strictly, the constraint binds a matmul's reduction (contracting) and output dimensions; the token/batch dimension streams through the array and needs no such alignment. But every model dimension listed above plays a reduction or output role in some GEMM of the forward or backward pass, so in practice they all want the alignment. The most-missed one is **head\_dim**: at head\_dim 128 or 64, the QK product in flash attention leaves the MXU ≥50% idle. Use **head\_dim \= 256** with proportionally fewer **Q and KV heads**, preserving both total widths. Ablations support iso-quality for this reshape.

&nbsp;

**Measured:** re-shaping Qwen3-30B MoE's attention from **32 heads × 128 to 16 × 256** (identical parameters and FLOPs) sped up end-to-end training and the gain *widens with context*:

&nbsp;

| context | head\_dim 128 | head\_dim 256 | gain |
| :---- | ----: | ----: | ----: |
| 8 K | 23.1% MFU | **28.1%** MFU | \+21% |
| 16 K | 23.3% | **30.8%** | \+32% |
| 32 K | 23.7% | **34.7%** | **\+46%** |

&nbsp;

MXU-aligned grouped-matmul tiles added a further \~+15% on the MoE GEMMs. A 2880-wide model (11.25 × 256\) pays padding that no tuning can  remove.

&nbsp;

**Precedent from GPU land:** NVIDIA revised Nemotron 3's final dims so the hidden dimension entering all-to-all was divisible by 512 (a DeepEP requirement); Microsoft's MAI-Base-1 ladder sets hidden size D \= L × 256/3 and rounds query heads up to multiples of 16 for tensor-parallel serving.

### **2\. One parallelism plan from 4 chips to multi-pod**

Pick dimensions so a single parallelism plan works at every scale: debuggable on 4 chips, efficient on one ICI slice, and scalable across pods with **only DP/FSDP/PP over DCN, never EP or TP** (Principle 10 covers the physical mapping). One constraint follows: the global batch must divide evenly over the batch-sharding axes of every mesh you plan to use.

Across DCN, both flavors of data parallelism work. Plain DP replicates the weights and syncs gradients once per step. FSDP shards the weights too, which saves HBM, at the cost of re-gathering them over DCN. Without gradient accumulation the two move comparable bytes and both can hide behind compute; with accumulation, every microbatch repeats FSDP's re-gather while DP still syncs once per step. So default to plain DP (slightly less traffic, one easy-to-hide sync) and switch to FSDP when the model needs the HBM saving; confirm either way with a short measured run. The same mechanism makes FSDP combine poorly with PP: the pipeline runs many microbatches per step, and each one repeats the re-gather, so pair PP with plain DP when HBM allows.

**Measured:** the DP-over-DCN path is near-free in practice: doubling to a second slice with data parallelism cost only \~1% of per-chip throughput.

### **3\. Static shapes everywhere**

JAX and XLA compile a program specialized to the exact input shapes and dtypes; a new shape triggers recompilation and stalls the run. Keep every hot-path shape static:

* Serve with a small set of fixed sequence-length buckets (each bucket compiles once); don't let request length set tensor shapes.  
* Keep data-dependent control flow out of the hot path; handle variable content with masking, not with shape-changing branches.  
* For MoE, prefer **dropless** routing (in MaxText: **sparse\_matmul=True,** with Megablox by default or **use\_tokamax\_gmm=True**). Both routing strategies are shape-static; the difference is **model quality**: capacity-factor drops the tokens that overflow its per-expert budget, which can degrade the model, while dropless keeps every token via ragged grouped matmuls. There is no speed penalty for choosing quality: the TPU ragged-GMM kernels make the dropless path efficient, while capacity-factor spends its compute on dense matmuls over padded capacity buffers. The dense-dropless fallback (sparse\_matmul=False) is a debugging baseline, not a benchmark.  
* Pack training sequences with proper attention masking so every batch fills its static shape (**packing=True** in MaxText, the default); padding tokens burn full FLOPs on a TPU.

Supporting evidence from MAI: capacity-capped MoE ablations can mislead even at low drop rates, so the MAI team converged on fully dropless, which the TPU ragged-GMM kernels provide natively. Kimi K3 goes a step further and designs the router itself for this property: quantile-based expert balancing gives fully balanced expert-parallel training "with static shapes and no host synchronization on the critical path".

## **Architecture choices (validate with ablations)**

### **4\. LatentMoE: shrink what moves, reinvest in expert capacity**

Expert parallelism sends every routed token between chips at full d\_model width. In Ironwood benchmarks that traffic was expensive enough that plain FSDP beat EP+FSDP. The structural fix is **LatentMoE**: compress each token from d\_model down to a latent dim ℓ before dispatch, run the experts entirely in latent space, and project back up after combine. Everything that moves between chips is then ℓ-wide instead of d\_model-wide. Adoption is broad: introduced by Nemotron 3, adopted by Microsoft's MAI-Base-1 (34.7B active / 962B total), and shipped by Kimi K3 as "Stable LatentMoE" at 2.8T total params (16-of-896 experts), the first open model at that scale.

Quantified design rules from Nemotron 3:

* **Two regimes, two bottlenecks.** Latency-bound serving (small batch) is dominated by reading expert weights (∝ d × m). Throughput-bound serving and training are dominated by all-to-all volume (∝ top-K × d, independent of expert width m). Expressive power scales roughly with the nonlinearity budget K × m.  
* **Compression factor d/ℓ ≈ 2–4×** cuts both bottlenecks by that factor. Reinvest the savings: scale total experts N and active experts K up by \~d/ℓ at constant cost. Nemotron's ablation: LatentMoE (ℓ=1024, 512 experts, top-22) beat standard MoE (d=4096, 128 experts, top-6) across every benchmark at identical active/total params.  
* **Route on the full-d representation**, compress after the routing decision; keep the gate, shared expert, and all non-routed compute at full d (MAI-Base-1 does exactly this).  
* **TPU specifics:** make ℓ a multiple of 256; dispatch in FP8 (halves payload again); keep the latent up/down projections in BF16 — they're a negligible fraction of step time. On a 3D torus, keep the EP group on one short physical axis.  
* **Consider a larger shared expert.** Shared-expert FLOPs are dense, communication-free, MXU-friendly, and FSDP-shardable, and they can be scheduled to **overlap with the all-to-all** (compute the shared path while dispatch/combine is in flight). Shifting part of the active-parameter budget from routed experts into a bigger shared expert trades some specialization for higher MFU and hidden EP latency; at decode, its weight reads amortize across the whole batch. Ablate the routed:shared split rather than copying GPU-tuned configs; the TPU optimum sits more shared-heavy. (Caveat: this applies to MoE-every-layer designs; the interleaved layout in Principle 5 gets a similar effect from its dense layers and typically drops shared experts.)

&nbsp;

**Measured** (our internal numbers, one 30B-class config): a 768-latent (2.67×) variant of a 30B / 128-expert MoE cut the on-device permute 2.8× and gained \+14% tokens/s end-to-end (322 vs 270 TFLOP/s per device). Past the down projection tokens are ℓ-wide, so all dispatch movement shrinks: the on-device permute and combine, not just the all-to-all under EP. For context, current MoE training recipes in [tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training) run FSDP or FSDP+TP rather than EP; EP becomes necessary when the per-device batch is too small to hide FSDP's weight all-gathers behind per-layer compute.

&nbsp;

**Recommendation**:

* Set ℓ by loss within the 2–4× compression band above.  
* Under FSDP, prefer fewer, larger experts. Per-expert GEMM group size shrinks as tokens × K ÷ N, so a large expert count starves the MXU; Nemotron's reinvest-into-more-experts rule belongs to the frontier-EP regime. Compress the token path first; expert-weight traffic is usually already hidden under FSDP.  
* At a larger scale, decide FSDP vs EP with a roofline computation. Latent compression cuts the all-to-all payload by d/ℓ and FP8 dispatch halves it again, which can make EP viable for models where standard dispatch is not. Verify the choice with a short profiled benchmark.

&nbsp;

### **5\. If quality-neutral, higher compute intensity is preferred (e.g. shared experts, larger feed forward dimension)**

Instead of making every layer a MoE layer, alternate: one high-sparsity MoE layer (MAI uses 8 active of 512 experts), then one small **dense** FFN layer. MAI tested this against the common alternative, medium-sparsity MoE in every layer, with total and active parameters matched and quality read from evaluation-loss scaling curves. At equal FLOPs the two layouts are equivalent; at equal training time the interleave clearly wins, because only half the layers dispatch tokens: the all-to-all events per forward pass are halved, a saving worth even more on a TPU torus than on NVLink. Make the first FFN layer dense as well (DeepSeek and Kimi practice); it stabilizes routing in early training.

The two options place the same ingredient differently: dense compute that runs for every token, with no routing or all-to-all dispatch, just large GEMMs with high arithmetic intensity. Pick one, not both. Interleaving halves the all-to-all events; every-layer MoE with a shared expert (Principle 4\) is the common open-source layout (DeepSeek-V3, Kimi K2). Ablate the choice at your target scale.&nbsp;

Larger `feed forward dimension` also increases compute per dispatched byte at fixed routing width; ablate it at matched quality and an explicit parameter/compute budget.

&nbsp;

### **6\. Low-precision as architecture, not afterthought**

Low-precision training is mixed-precision training: quantize the large GEMMs and keep a small set of operations in higher precision. A good starting set: every softmax input (attention scores, router scores, output logits), the router gating, norms, embeddings, and the final layers. Beyond that there is no fixed rule: watch the training loss curve, and promote an operation to higher precision only when the curve turns unstable or the loss degrades.

Architecture shrinks the high-precision set: attention-logit bounding (QK-norm, or Kimi's weight-space QK-clip) and output-logit control (z-loss or logit caps) bound dynamic ranges, which lets cheap static or coarse scaling hold quality. This matters more on TPU than on GPU, where fine-grained hardware block scaling can rescue unbounded ranges.

**Block-scaled (sub-channel) quantization costs VPU time on v7x and earlier.** The MXU applies no per-block scales in hardware, so block scaling runs on the VPU, and its cost relative to the matmul is set by the MXU:VPU throughput ratio, about 240:1 on v7x (Principle 9); finer blocks mean more VPU work per MXU FLOP. Use blocks of 512 or larger on v7x for FP8, and rely on the range-bounding above to keep coarse blocks quality-safe. The [Qwen3.5-397B on Ironwood optimization playbook](https://developers.googleblog.com/systems-engineering-playbook-optimizing-qwen-35-397b-moe-on-ironwood-tpu7x/) made the same move, switching to 512-element subchannel scaling for FP8 to remove VPU register spills and stalls.

**TPU v8 supports block scaling in MXU for FP8 and FP4, so the v7x VPU-driven ≥512 block-size recommendation does not apply.**

**Where FP8 pays (tpu-recipes, Ironwood):** dense models gain \~1.5×, MoE \~1.2×, because routing and communication dominate the MoE step and GEMM-only FP8 does not reduce those costs. Neither number is a ceiling: the more matmul-dominated the optimized step becomes, the more every precision drop pays, so shrink the plumbing first (Principles 4 and 9–10).

**Serving can go lower than training.** Appendix A.7 covers post-training quantization; protect the same high-precision set and the aggressive step survives with minimal quality loss. Precision keeps falling (FP4-class pretraining is already demonstrated at scale), so range-bounding baked in now makes the next drop a config change, not a redesign.

**On MoE, lower expert precision first.** Expert GEMMs hold most of the parameters and FLOPs, sit behind the router's high-precision gate, and tolerate lower precision than attention. That makes FP4 expert compute an option on supported training hardware, not just serving (Nemotron 3's NVFP4 training recipe is the precedent). On v7x, distinguish FP4 storage/numerics from the kernel’s actual arithmetic format; validate conversion overhead and block size rather than assuming native FP4 MXU throughput. At serving time Kimi K2's thinking model already ships INT4 experts (quantization-aware trained) with the rest kept higher.

### **7\. Hybrid attention for long context: pick the TPU-mature variant**

Full attention on every layer is what makes long context expensive. The field has converged on hybrids that keep a few full-attention layers and make the rest cheaper, in three families:

* **Local/global interleave:** most layers use sliding-window attention, a few stay global (Gemma-3 and MAI-Base-1 at 5:1 with window 512; MAI drops positional encoding on the global layers, which helps length extrapolation).  
* **Linear-attention hybrids:** most layers use a constant-state recurrent form: Mamba-2 (Nemotron 3), Gated DeltaNet (Qwen3-Next), Kimi Delta Attention (Kimi K3 at 1M context). Constant state per sequence, no RoPE extrapolation issues, and decode cost independent of context length.  
* **Compressed or sparse attention:** keep attention everywhere but shrink what it reads: latent-compressed KV (DeepSeek’s MLA) and learned sparse attention over it (the DeepSeek-V3.2/V4 line).

&nbsp;

TPU guidance: the **local/global interleave is the safe default**: sliding-window attention maps directly onto Splash attention with VMEM-friendly tiles (make window sizes multiples of the kernel block size). Linear-attention hybrids are the more aggressive option; their scan kernels are less mature on TPU than the attention stack (no fused training kernels yet, so the recurrent layer can dominate the profile), so budget Pallas kernel work before committing.

**KV-cache implications:** Hybrid attention also cuts the KV cache substantially, which pays twice at decode: HBM capacity caps the batch and KV reads set bandwidth (Principle 12). Windowed layers cap KV at the window size; linear-attention layers replace KV with a fixed-size state (the largest cut); latent-compressed KV shrinks each entry. Size the cache at target batch × context before picking the family.

**Paper FLOP savings need a kernel that realizes them.** Cutting attention FLOPs also cuts arithmetic intensity: the kernel still streams Q, K, and V, so a windowed or linear-attention layer can run memory-bound and land far off its FLOP-count promise, even while the mature global-attention kernel sits near its roofline. Pressure-test the cheap-attention kernel at your window, context, and head shape before committing the architecture: budget the savings from measured kernel time, not FLOP counts, and expect kernel tuning or optimization work to close the gap.

### **8\. Speculative decoding as a design input (MTP heads or another drafter)**

At low concurrency, decode is often memory-bandwidth-bound (Appendix A.4). A drafter proposes tokens and the main model verifies them together, so speculative decoding can amortize each weight read over multiple accepted tokens. Use MTP heads (or a small draft model) to improve decode latency / throughput.

**MTP heads are the design-time option.** They give a denser training signal and a built-in drafter without a separate model. [DeepSeek-V3](https://arxiv.org/html/2412.19437v2#S5.SS4.SSS3) reports 85–90% acceptance of the extra drafted token; [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) and [GLM-4.5](https://arxiv.org/html/2508.06471v1#S2) also use MTP. In an 8B-active MoE ablation, [Nemotron 3](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf#page=5) reports \~2.4% average relative benchmark improvement and \~97% acceptance on the first two drafted tokens. Keep MTP layers in high precision.

**Measure committed tokens/s at the target batch, not acceptance alone.** Include drafting, verification, and rollback costs. Tune draft length to concurrency: the benefit generally shrinks as decode becomes compute-bound.

### **9\. Minimize exposed VPU time**

The MXU does matmuls; norms, activations, elementwise ops, and dispatch math run on the vector processing unit (VPU) and are usually memory-bound. The objective is not to remove quality-relevant ops but to minimize the VPU time that sits exposed on the critical path. Three buckets, three levers:

* **Remove what buys no quality:** gratuitous reshapes and transposes, layout format copies, and tiny per-token ops that never amortize an HBM round trip.  
* **Fuse what stays:** quality-relevant ops (norms, activations, routing math) are kept, in fusable forms. Use one norm style throughout (RMSNorm) and standard fusable activations (SwiGLU is fine); XLA folds elementwise work into adjacent GEMMs, and fused attention kernels run softmax under the matmuls. Standalone VPU ops between GEMMs run largely sequentially on the TensorCore, so fusion, not scheduling, is the lever.  
* **Offload dispatch data movement to SparseCore:** sorts, gathers, and scatters belong on SparseCore (Principle 10), the engine that genuinely runs beside the TensorCore.

**Size VPU work against the hardware ratio.** On v7x the FP8 MXU-to-f32 VPU throughput ratio is roughly 240:1, so within a fused kernel a VPU op can hide behind an adjacent MXU op only when its execution fits within the MXU compute window; this ratio helps set a practical minimum block size for block-scaled quantization (Principle 6), together with register pressure and memory traffic.

## **Systems mapping (no model changes)**

### **10\. Map model axes onto the physical mesh**

The rule that is right 99% of the time: **give every parallelism axis a physical network axis, with the most communication-intensive sharding on the fastest link** (D2D → ICI torus → DCN, fastest to slowest). Mismatches show up directly as exposed communication time: TP spanning chips, EP or TP crossing DCN, or a logical mesh whose factors don't divide the physical topology. Within that hierarchy:

&nbsp;

* **TP \= 2 max for training**, mapped to the dual chiplets. Wider TP needs MLP dim \> \~46k (TP4) to clear the 11.5k arithmetic-intensity bar; almost nothing qualifies. **Inference is different:** decode is bandwidth-bound, so TP there shards weight *reads* rather than hiding compute — TPU vLLM recipe serves Qwen3-Coder-480B-A35B with TP8 \+ expert parallelism.&nbsp;  
* **FSDP as the default**; per-device batch large enough to hide the weight all-gather.  
* **EP on one short torus axis; PP/DP across pods over DCN.** The exception: at small per-device batch, FSDP can no longer hide its weight all-gathers, and PP inside the pod can win, since it moves only stage-boundary activations. Decide with a roofline computation, confirmed with a short profiled run (the same method as FSDP vs EP, Principle 4).  
* **CP at the minimum degree that fits memory.** It comes in three flavors; pick by what each exposes.  
  * **All-gather CP:** simplest, but the all-gathers can sit exposed, and it needs load-balanced (zigzag/striped) sequence order: in plain order, causal masking leaves the chips holding later chunks idle. MaxText's `context_parallel_load_balance` provides this.  
  * **Ring CP:** hides the collectives, but needs the same load balancing plus packing treatment, and its chunking costs Splash-kernel efficiency.  
  * **A2A CP (Ulysses):** no load imbalance, but exposes the all-to-all and is hard-capped by the head count.  
* **SparseCore offload on:** Offloading async collectives to SparseCore can produce a 15–22% step-time reduction (seen in tpu-recipes benchmarks). The recipe flags offload all-gather, reduce-scatter, and all-reduce (including 2D/3D variants) plus embedding-style gathers to SparseCore; it's XLA flags, not model surgery. If your model has large embedding tables or other sparse-access components (recsys-style), SparseCore is also the native engine for those lookups.

### **11\. Tune the memory pipeline**

Use **Pallas** to make HBM→VMEM prefetch and buffering explicit, overlapping transfers with compute while fitting tiles, intermediates, and buffers in VMEM. [Pallas pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html). Two empirical knobs control the memory pipeline. The XLA flag `xla_tpu_scoped_vmem_limit_kib` sets how much of the 64 MiB VMEM kernels may claim as scratch: raising it buys bigger tiles and fewer memory stalls, at the cost of weight-prefetch headroom. Kernel block sizes are the second knob: search them with `tune-jax`, since defaults are usually suboptimal and **block-size optima are shape-dependent**. For reference, the DeepSeek-V3 Ironwood recipe runs scoped VMEM at the full 65,536 KiB, Splash attention blocks at 2048, custom remat, and a two-stage all-gather for FSDP-sharded expert weights. The recipes are worth mining as tuned starting points, not just launch scripts.

### **12\. Co-design for prefill/decode disaggregation, and know the TPU serving stack**

Prefill is typically **compute-bound**; low-concurrency decode is typically **memory bandwidth-bound**. They often benefit from different slice shapes, sharding, and batch sizes. Tune each phase against its latency/throughput trade-off: time to first token (TTFT) for prefill, and time per output tokens (TPOT) for decode. **Disaggregation** lets each phase use a pool suited to those requirements.&nbsp;

Know what the serving stack already assumes. From the vLLM-on-Ironwood recipes: FP8 KV cache, **KV `block-size=256`** (the MXU alignment rule reaches into the paged-attention cache layout), and parallelism by regime (TP8 plus expert parallelism for the \~400B MoEs, TP2 on the chiplet pair for the 20–32B models).&nbsp;

&nbsp;

## **Which principles are most beneficial**&nbsp;

If you adopt only a handful: 256-aligned dimensions (1), static shapes with dropless routing (3), one parallelism plan mapped to physical links (2, 10), wall-clock ablations on a tuned stack (Appendix A.6), and reference targets to judge the result (Appendix A.8). Those are cheap, quality-safe, and cover the largest measured gaps. For MoE models add LatentMoE (4): it needs an ablation, but it reduces the routing and communication volume that dominates MoE steps, paid off in our training measurements, and shrinks the expert weight reads that bound decode. Beyond those, the regime you are in decides what to reach for next:

&nbsp;

| Regime | Reach first for | Why |
| :---- | :---- | :---- |
| High expert count (over 100 experts) | 4, 5, 9, 10 | routing and dispatch dominate the step: compress the token path, pair the MoE with dense compute, offload dispatch to SparseCore, keep EP on one torus axis; small per-expert GEMMs starve the MXU, so prefer fewer, larger experts under FSDP |
| Long context (over 64K) | 7, 11 (or App. A.3), 1, 10 | attention cost and activation memory take over: hybrid attention is the main lever, remat policy second; head\_dim 256 gains widen with context at fixed parameters, and the CP flavor decides what communication sits exposed |
| Very large models (over \~100B, or any multi-pod run) | 2, 10, 4, 6, 12 | communication becomes the binding constraint, and the parallelism plan decides how much of it sits exposed: only DP/PP over DCN, EP viability by roofline, FP8 pays more as the step gets matmul-dominated, and the serving ladder should be planned, not retrofitted |
| Serving | 8, 12, App. A.4, App. A.7, 7 | at low concurrency, decode is bound by weight reads, plus the layer count since layers run one after another: size active params to the decode roofline, add MTP/speculative decoding, quantize further. At high concurrency or long context, the KV cache takes over: shrink it (hybrid or compressed attention, FP8 KV) |

The regime labels are rough headings, not thresholds to compute; to find your regime by measurement, use the Diagnosing your own runs section at the end of the doc.

&nbsp;

## **Diagnosing your own runs**

The principles above are based on profiles captured with public tools (`jax.profiler`, which in MaxText is enabled by the flag **profiler=xplane**), so you can verify each one on your own model. Open the trace in the [XProf / TensorBoard profiler](https://openxla.org/xprof) and check:

&nbsp;

| \# | number | healthy | if off, apply |
| :---- | :---- | :---- | :---- |
| 1 | exposed (un-overlapped) communication, share of step | a few % | try **overlap first**: scheduler \+ SparseCore-offload flags (Principle 10). If still exposed, there is likely **too little compute per byte moved** (the arithmetic-intensity bar of the link it crosses) and no schedule can hide it; cut communication volume via sharding choice or compression (Principles 4, 10\) |
| 2 | backward:forward wall-clock ratio | ≈ 2-3 | 2 means no remat, ≈3 means full remat; anywhere in that band is a deliberate compute-for-memory trade (remat frees HBM for larger batches, and is unavoidable at large scale). Act when the ratio is higher than your HBM pressure requires: store more activations where memory allows, sharding more if HBM is tight (Principle 11). A ratio remat can't explain (past \~4) points to the backward itself: exposed collectives or slow backward kernels |
| 3 | large GEMMs, % of peak FLOP/s | 80–95% | the big matmuls should individually run near the chip's peak; lower means the MXU is partly idle, usually because a model dimension is not a multiple of 256 (Principle 1\) or kernel tile sizes are off (tune them, Principle 11\) |
| 4 | non-matmul share of step time (XProf op profile, grouped by category) | under \~15% for dense; under \~30% on an MoE | above that, non-matmul work is eating the step. In the profile it appears as standalone elementwise fusions (the VPU work) and, on MoE, the routing ops: sort, argmax/one-hot, cumsum, gather/scatter. Fix by fusing elementwise ops into the surrounding GEMMs (Principle 9); on MoE, also shrink the routing overhead itself: latent compression, fewer/larger experts, dispatch kernels (Principles 4, 9\) |
| 5 | engine running sorts/gathers/scatters | SparseCore, in most tuned configs | if they run on the TensorCore, the SparseCore-offload flags are off, or a runtime release changed their defaults (Principle 10). Offload usually wins (15–22% in tpu-recipes benchmarks) but not always; confirm the direction with a short measured run |
| 6 | step-time variance after warmup | steady, near-identical step time | the compiled TPU program is deterministic, so the same step should take the same time; jitter means steps are waiting on something outside it, usually the input pipeline (data loading, host CPU), not the model |

&nbsp;

These ranges are triage thresholds, not universal constants. What is achievable depends on the model, the scale, and the sharding: a sharding plan that moves more bytes per FLOP has a higher healthy exposed-comm share, and a small model's GEMMs will not reach a big model's % of peak. So judge your numbers against a tuned reference of similar scale and sharding (Appendix A.8), then ask whether the remaining gap is in execution (fixable with flags and kernel tuning) or inherent to the design (fixable only by changing the model or the sharding).

&nbsp;

**Roofline a kernel before tuning it.** The table above triages the whole step; for a single kernel or op, the roofline says what speed is even possible. Arithmetic intensity is FLOPs per byte moved over the channel that feeds the op, usually HBM for a kernel; the same logic applies to a collective over ICI (the exposed-communication row above). The op's FLOP/s ceiling is intensity × that channel's bandwidth, up to the chip's compute peak. Judge the op profile's measured FLOP/s against that ceiling, not against peak: a memory-bound op at its ceiling is already as fast as the hardware allows, and the lever is moving fewer bytes (fusion, lower precision, smaller KV or latent widths), not tile search. `jax.experimental.roofline` computes the static FLOPs and bytes per function; Principle 7's attention pressure-test is this analysis in action.

&nbsp;

**Budget HBM before sweeping.** HBM use has a fixed part and a scaling part. The fixed part is weights, optimizer state, and gradients: ≈16 bytes/param at MaxText defaults, divided by every parallelism degree that shards the weights (FSDP, TP, EP, PP); plain data parallelism replicates instead. If the fixed part doesn't fit, it can move to host memory (**parameter\_memory\_host\_offload**, **optimizer\_memory\_host\_offload**). The scaling part is activations, which grow with batch × sequence. When a config doesn't fit, apply the levers in the order that worked for us: shard more (FSDP/EP degree), then adjust the remat policy, then cut per-device batch, with host offload last (it usually loses performance, Appendix A.3).

&nbsp;

**Run optimization as a program.** Change one knob per run; 20-30 steps is enough for a steady-state step time. After each change, capture a new profile and verify the *mechanism* in it, not just the speedup: the remat re-runs are visibly gone, the dispatch ops actually moved engines. Keep the negative results too; they close questions permanently. And gate every perf change on numerics with a fixed-seed loss-parity run: one of our "wins" turned out to be compiler-version-specific and vanished on the next runtime release, and the parity gate is what caught it.

## **Putting it together: a \~100B-class LatentMoE reference config for Ironwood**

| Component | Choice | Rationale |
| :---- | :---- | :---- |
| d\_model | 6144 (24×256) | MXU alignment; wide-not-deep |
| Layers | 48 | Divisible by PP degrees; 5:1 local:global attention pattern fits |
| Attention | head\_dim 256, 24 Q heads, QK-norm; 5:1 sliding-window(512):global, NoPE on global | Full MXU utilization; length extrapolation |
| KV compression | GQA with 4 KV heads \+ FP8 KV cache (or MLA-style 512-dim latent for deeper compression) | Decode bandwidth \+ HBM fit |
| FFN layout | Alternate dense FFN (d\_ff 8192\) with LatentMoE; first FFN dense | halved all-to-all events; routing stability |
| LatentMoE | ℓ \= 2048 (3× compression), 128 experts, top-8, expert d\_ff 4096, FP8 dispatch, route on full d; shared expert sized to overlap with the all-to-all | K×m nonlinearity budget reinvested per Nemotron rules; EP latency hidden |
| MTP | 1–2 draft heads, BF16 | \+quality, **speculative decoding** |
| Precision | FP8 E4M3/E5M2 (RNE); sensitive-layer map in BF16/FP32; last layers high precision | Train-serve consistency; FP4-ready |
| Training layout | FSDP × EP8 (one torus axis) × TP2 (D2D); SparseCore offload; zero-init attention outputs; global-batch load balancing; dropless via Megablox |  |
| Serving | Same FP8 weights \+ KV; prefill/decode disaggregated; \~600-token/chip decode roofline target met via sparsity \+ MTP |  |

&nbsp;

## **Appendix A. Supporting design and evaluation topics**

### **A.1. Stay in the conventional aspect-ratio band; explore wider-not-deeper within it via ladder ablations**

For a fixed parameter budget, wider-shallower runs faster: larger matmuls saturate the MXU, fewer sequential ops cut launch overhead and decode latency. The quality side must be established per model: pretraining loss is only weakly shape-sensitive **within the conventional aspect-ratio band** (loss stays nearly flat across a \~40× aspect-ratio change at fixed parameters, Kaplan et al.; depth efficiency saturates at large width, Levine et al.), but downstream quality is more shape-sensitive than loss and can favor depth (Tay et al.). So treat width as a lever to explore, not a rule to adopt: **run param-matched ladder ablations** gated on downstream evals, and stay within the established band. Precedent: MAI pins a conventional aspect ratio (D \= L × 256/3) across its ladder because it yields hardware-friendly widths, rather than pushing wide. Make layer count divisible by plausible pipeline degrees (48, 64, 72...).

**Measured (directional):** a wider-shallower 30B variant ran at 31.4% vs 29.3% MFU; confirm with a param-matched ladder ablation before adopting.&nbsp;

&nbsp;

### **A.2. One knob (layer count) scales the family; ratios stay TPU-friendly**

One knob, the layer count L, defines the whole model family: width and heads derive from fixed ratios that land on hardware-friendly values, and every ablation runs as a ladder of sizes at constant tokens-per-parameter. The TPU-friendly ratios: width a multiple of 256; head count divisible by your TP degrees (TP splits attention by head; 2 for chiplet TP, 8–16 for serving, MAI rounds to 16); expert count divisible by candidate EP degrees (8, 16, 32); layer count divisible by the planned pipeline degree (48, 64, and 72 are example layer counts). Then every ablation is also a shape-legal production candidate.

&nbsp;

### **A.3. Remat-friendly blocks**

Rematerialization can cost \~25–30% extra FLOPs, depending on the checkpoint policy and is the main lever against activation OOM at long sequence lengths (or large per-device batch size). Design blocks so the expensive-to-recompute tensors (attention outputs, expert outputs) are few and checkpointable, with cheap elementwise work between them — this makes intermediate remat policies (`save_qkv_proj`, `save_out_proj`) effective instead of forcing `full` remat. MaxText's `custom` remat policy covers the rest.

**Measured: keep activations in HBM by default.** Host offload moves data over a link far slower than HBM; in our tests, keeping attention outputs in HBM beat every offload variant by **\+8–19%**. When weights and optimizer state crowd HBM, offloading a small chosen set of activations can still pay, and the tpu-recipes configs do exactly that (the Ironwood 405B recipe offloads the decoder-layer input; Trillium and v5p recipes add the QKV projections). Revisit per model and chip with a measured run. Diagnostic: backward:forward time around 2 without remat and around 3 with full remat are rough compute-based references, not wall-clock ceilings. A LatentMoE bonus: anchoring checkpoints on the compressed latent instead of the full-width context freed \~25 GB/device of batch headroom.

&nbsp;

### **A.4. Size the model to your real serving concurrency**

Low-concurrency decode is typically memory-bandwidth-bound. The ideal dense FP8 weight-only crossover is \~300 tokens sharing each weight read; compare the batch seen by a weight shard, not replica concurrency divided by TP chip count. One model serves a range of concurrencies and contexts; size it for the demanding end (latency-sensitive traffic at your longest supported context), since high-concurrency traffic is more compute-bound and forgiving. Before committing the architecture, do two pieces of arithmetic:

**Per-token latency floor \= per-chip weight bytes ÷ HBM bandwidth.** Each decode step streams the chip's resident weights: for a dense model, its share under TP; for a MoE, the stream is the union of expert weights selected by the batch, plus dense/shared weights. It approaches the full model as expert occupancy rises; exceeding the sparsity ratio does not guarantee every expert is touched. Example: if the batch touches all experts and placement is balanced, Qwen3.5-397B in FP8 over 4 chips streams \~100 GB per chip, a \~13.5 ms floor before counting KV reads; widening the TP/EP unit buys latency, paid in interconnect traffic.

**Memory-limited concurrency ≈ HBM left after weights, runtime buffers, and workspaces ÷ allocated state per sequence** at your target context. This, or your latency target, caps the batch that shares each weight stream; per-token cost is streamed bytes ÷ batch. The expert-weight stream grows with the union of experts selected by the batch; use observed occupancy and actual per-rank KV/state allocation for this budget. The levers are the byte levers: FP8 weights and KV (4-bit halves again, Appendix A.7), TP/EP to spread the stream, small KV to raise the concurrency ceiling (Principle 7), and MTP/speculative decoding, which gets several tokens out of each weight read and acts like a concurrency multiplier (Principle 8). If your concurrency will be small, design fewer bytes (fewer active params, or a smaller model); don't plan around a batch size you won't have.

&nbsp;

### **A.5. Train-big-serve-small as a planned path**

Frontier deployments serve a ladder of sizes, not one model. The top tier is usually the trained flagship itself, typically a MoE, served quantized but otherwise unchanged (DeepSeek-V3, Kimi K3, Qwen3-235B). The lower tiers are distilled derivatives, often still MoEs (Qwen3-30B-A3B). Two consequences:

* **Design the flagship for serving directly** (Principle 12 and Appendices A.4 and A.7).  
* **Run every tier through the shape rules** (Principle 1 and Appendices A.1–A.2), including tiers pruned out of the teacher (Llama 3.2, Minitron). The 256-alignments matter *more* at small scale, where padding wastes a larger fraction.

&nbsp;

### **A.6. Measure ablations in wall-clock, not FLOPs**

FLOP counts and wall-clock rank architectures differently, because kernel maturity varies by op type. MAI's ablation methodology tracks the two separately, and the rankings flip: MoE-every-layer looked fine on FLOPs but lost clearly on time. So run architecture ablations with the real stack (Splash attention, Megablox, SparseCore offload, tuned tiles via `tune-jax`) and compare time/cost at matched quality. Use step time for equivalent systems changes at a fixed workload. The baseline must also be a ***tuned*** **stack**. Example: systems work alone (SparseCore offload, MXU-aligned tiles, remat policy) took an open weight 30B MoE from **12% to 23% MFU** in our tuning, before any model change; an ablation run against the untuned baseline mis-ranks the levers.

&nbsp;

### **A.7. Quantize further for serving**

Training precision is not the serving floor. Decode reads the weights used by the batch and reuses them across its tokens (Appendix A.4), so each halving of weight bits approximately halves weight-read bytes before metadata and conversion overhead: FP8 served weights are the train-consistent default (Principle 6), and weight-only INT4 / W4A8 / FP4 post-training quantization (PTQ) is a legitimate further step, approximately halving stored weight bytes again versus FP8. It needs no model change or retraining. Guidance:

* **Match the mode to the regime.** Small-batch, latency-bound decode is dominated by weight reads, so weight-only 4-bit pays most. Large-batch, throughput-bound serving is compute-bound, so activations matter too; W4A8 or plain FP8 fits better.  
* **Reuse the training sensitive-layer map** (Principle 6). Keep in high precision the same layers you protected during training. The range-bounding choices that made FP8 training cheap (QK-norm or QK-clip on attention logits; z-loss or logit caps on the output logits) also let aggressive quantization survive with little quality loss.  
* **Quantize the KV cache too.** In the long context, KV reads rival weight reads; FP8 KV is already standard in the Ironwood serving recipes (Principle 12).  
* **Calibrate on serving-like traffic.** Static activation scales are frozen from a calibration run; if that data doesn't match production traffic (domain, language, sequence length), real activations clip or waste range.  
* **Gate on downstream evals, not perplexity alone.** Quantization damage is task-dependent; validate on the tasks you actually serve.

&nbsp;

### **A.8. Reference performance targets**

Use published Ironwood recipe numbers as sanity-check targets for your own runs (TFLOPs/sec/chip, MaxText; divide by the chip peak, 2,307 BF16 or 4,614 FP8, for MFU):

| Model | BF16 (TFLOP/s/chip) | FP8 (TFLOP/s/chip) |
| :---- | :---- | :---- |
| Llama-3.1 70B (8k) | 1,207 | 1,854 |
| Llama-3.1 405B (8k) | 1,261 | 1,927 |
| Llama-3.1 70B (128k) | 898 | 1,014 |
| DeepSeek-V3 671B (4k) | 608–613 | 731–743 |
| Qwen3-235B-A22B (4k) | 630 | 703 |
| GPT-OSS-120B (8k) | 330 | — |

Two lessons in this table. First, the dense–MoE MFU gap is large: sparsity buys quality per FLOP but pays an MFU tax in routing, communication, and ragged shapes; budget accordingly. Second, the gap *within* MoEs matters: GPT-OSS-120B's many-small-experts, head\_dim-64 design lands at 330 TFLOPs, roughly half of DeepSeek's, empirical support for fewer/larger experts and 256-aligned head dims (Principles 1 and 4).

## **Appendix B. Inference hardware-software co-design**

Evaluate the architecture and serving configuration together at the intended prompt lengths, output lengths, concurrency, and latency targets. Compare quality and goodput—completed requests meeting both time-to-first-token (TTFT) and time-per-output-token (TPOT) targets per second—on the same compute allocation. The following are supporting recommendations, not additional principles.

| Choice | Guidance and mechanism | What to measure |
| :---- | :---- | :---- |
| **Attention, KV format, and kernel layout** | Keep the intended `head_dim = 256` change paired with proportionally fewer Q and KV heads, preserving their total widths and the iso-quality ablation result. Then check the *physical* KV allocation after sharding, packing, and padding. Tune KV pages and attention compute blocks separately; neither is determined solely by the MXU's 256-wide dimensions. Prefer fused KV updates and phase-appropriate attention kernels. Quantized KV must save actual bytes and preserve quality. TPU RPA explains these layout and fusion effects. | Allocated KV bytes/token, attention and KV-update time, padding, layout conversions, long-context quality. |
| **Attention parallelism and expert parallelism** | Select attention TP/DP separately from EP. A large expert pool need not imply equally wide attention TP; account for KV replication and small local head counts. Compare wider sharding with more serving replicas at a fixed chip budget: replica count determines the batch available to amortize weight reads. [Google’s Ironwood deployment](https://developers.googleblog.com/systems-engineering-playbook-optimizing-qwen-35-397b-moe-on-ironwood-tpu7x/) uses attention DP with EP to avoid duplicating KV heads. | Per-replica concurrency, KV replication, collective latency, total goodput/chip and tail latency. |
| **MoE occupancy and load balance** | Size expert width, count, and routing for the actual decode batch. Per-token active parameters do not describe the union of expert weights read by a batch. Profile skew as well as average tokens/expert; overloaded ranks set latency. Rebalance placement or replicate hot experts when HBM allows, without changing router semantics. [DeepSeek-V3](https://arxiv.org/pdf/2412.19437) uses redundant experts for deployment load balance. | Distinct experts touched, tokens/expert distribution, busiest-rank time, dispatch/combine bytes and expert-weight bytes. |
| **Batching and prefill/decode placement** | Use a small set of prewarmed compiled token/slot buckets. Tune continuous batching and prefill chunk size jointly: long prefills can delay ongoing decode, while tiny chunks waste compute. Compare colocated chunked prefill against disaggregation with separately chosen pool sizes and shardings. Include KV handoff, any layout conversion, and queueing in the comparison. [Sarathi-Serve](https://arxiv.org/abs/2403.02310) and [DistServe](https://arxiv.org/abs/2401.09670) establish the scheduling and placement mechanisms; their GPU speedups are not TPU forecasts. | TTFT and time-per-output-token percentiles, queue time, transfer bytes/time, prefill/decode pool utilization, goodput. |
| **Prefix reuse and routing** | Reuse exact matching prefix KV where the serving backend supports it. Route related requests using both cache locality and load, so queueing does not erase saved prefill. Prefix reuse primarily reduces repeated prefill; it does not by itself eliminate reading that context during decode. [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) and [llm-d routing](https://llm-d.ai/docs/architecture) describe these mechanisms. | Reused prefix tokens, TTFT including queueing, eviction/recomputation, active KV capacity, load skew. |
| **MTP/speculative decoding** | Tune draft length for the whole draft–verify–commit cycle, at the target batch. Acceptance alone is insufficient: extra verification work, drafter cost, KV rollback/commit, and scheduling affect the gain. Measure committed output tokens per second and validate the sampler; distribution preservation requires the appropriate verification/correction procedure. [Speculative decoding](https://arxiv.org/abs/2211.17192). | Accepted prefix length, committed tokens/cycle, complete cycle time, output distribution/quality and goodput. |

&nbsp;

**Two useful budget checks.** For conventional KV with equal K/V head dimensions and precision, the logical cache size is `M_KV = 2 * s_KV * sum_{r,l} (S_{r,l} * n_{KV,l} * d_{head,l})`, where `s_KV` is bytes per element and `S_{r,l}` is the cached token count for request `r`, layer `l`. The proportional head-count reduction preserves this logical budget. Physical allocation also includes paging, padding, quantization metadata, and replication; latent attention and recurrent states require their own state accounting. For a toy model with `E` equally likely experts, `K` distinct routes per token, and `B` independently routed decode tokens, the expected expert count touched in one MoE layer is `E_touched = E * [1 - (1 - K/E)^B]`. This is a derived occupancy estimate, not a routing assumption to impose on the model. Use measured occupancy and weight reads for real traffic. LatentMoE's save-and-reinvest condition `K_new * d_latent <= K_old * d_model` preserves or reduces routed activation payload at equal precision; expert-weight reads and compute are separate budgets. The iso-quality statements above refer to the authors’ coordinated-reshape and LatentMoE ablations. The linked RPA paper supports kernel/layout guidance; its fixed-head-count head-dimension comparison is a different experiment.

&nbsp;

## **Sources**

* Google Cloud, *TPU7x (Ironwood) performance optimizations* — docs.cloud.google.com/tpu/docs/ironwood-performance  
* Google Developer Forums, *Optimizing frontier model training on TPU v7x (Ironwood)*, Mar 2026  
* JAX ML, *How to Scale Your Model* (scaling book) — jax-ml.github.io/scaling-book  
* AI-Hypercomputer, *tpu-recipes* (Ironwood training \+ inference recipes) — github.com/AI-Hypercomputer/tpu-recipes  
* NVIDIA, *Nemotron 3: Efficient and Open Intelligence*, arXiv:2512.20856 — LatentMoE, MTP, NVFP4 recipe  
* Microsoft AI, *MAI-Thinking-1: Building a Hill-Climbing Machine* — microsoft.ai/pdf/mai-thinking-1.pdf — scaling ladders, EG\_Time, interleaved sparsity, precision map  
* Ironwood profiling program (open 20–30B models — Qwen3-30B-derived LatentMoE, GPT-OSS-20B — plus dense controls; single TPU slice, bf16)  
* Tools: [XProf / TensorBoard profiler](https://docs.jax.dev/en/latest/profiling.html) (traces, op profile), [`tune-jax`](https://github.com/rdyro/tune-jax) (empirical kernel block search), [`jax.experimental.roofline`](https://github.com/jax-ml/jax/tree/main/jax/experimental/roofline) (static FLOPs/bytes per function), [MaxText](https://github.com/AI-Hypercomputer/maxtext) (reference configs for every knob named here).  
* Shape vs quality (Appendix A.1): Kaplan et al., *Scaling Laws for Neural Language Models*, arXiv:2001.08361 (Fig. 5 — loss ≈ flat across \~40× aspect-ratio change at fixed params); Levine et al., *The Depth-to-Width Interplay in Self-Attention*, arXiv:2006.12467 (depth efficiency saturates ∝ log width); Tay et al., *Scale Efficiently*, arXiv:2109.10686 (downstream quality is more shape-sensitive than loss)

&nbsp;

&nbsp;

&nbsp;

&nbsp;
