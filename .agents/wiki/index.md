# MaxText Model Bring-up & Engineering Wiki

Shared engineering playbooks, parity diagnostics, and implementation recipes distilled from MaxText model bring-ups across Linen (`src/maxtext/layers/`) and NNX M3 (`src/maxtext/m3/`).

Load **only** the page matching your active phase or blocker to keep context lean. Read [`PROTOCOL.md`](PROTOCOL.md) **only** when distilling or editing wiki pages.

---

## Model Archetype & Reference Precedents

Match your target HuggingFace architecture during Phase 1 to inspect the closest existing MaxText implementation, parity test, and pre-flight wiki pages:

| Architectural Trait | Closest MaxText Reference & Test | Pre-Read Wiki Pages |
| :--- | :--- | :--- |
| Dense GQA + RoPE (`Qwen3`, `Llama`) | [`qwen3.py`](../../src/maxtext/models/qwen3.py), [`m3/models/qwen3/`](../../src/maxtext/m3/models/qwen3/__init__.py) ([`qwen3_comparison_test.py`](../../tests/unit/m3/qwen3_comparison_test.py)) | — |
| Dense-prefix + Routed/Shared MoE + `e_score_correction_bias` (`DeepSeek`, `GLM-5`) | [`deepseek.py`](../../src/maxtext/models/deepseek.py), [`m3/models/glm5_3_flash/`](../../src/maxtext/m3/models/glm5_3_flash/__init__.py) ([`glm5_3_flash_moe_test.py`](../../tests/unit/m3/glm5_3_flash_moe_test.py)) | [`troubleshooting/hbm_verification.md`](troubleshooting/hbm_verification.md) |
| Hybrid Linear Attention (GDN) + Softmax Cycle + Routed MoE (`Qwen3-Next`, `Qwen3.8`) | [`qwen3.py`](../../src/maxtext/models/qwen3.py) ([`qwen3_next_vs_reference_test.py`](../../tests/unit/qwen3_next_vs_reference_test.py)), [`m3/models/qwen3_8_flash_next/`](../../src/maxtext/m3/models/qwen3_8_flash_next/__init__.py) | [`troubleshooting/hbm_verification.md`](troubleshooting/hbm_verification.md) |
| Pure-NNX Routed MoE + `FrozenTopK` + Compound `EP×TP×FSDP` (`GraniteMoE`) | [`m3/models/granitemoe/`](../../src/maxtext/m3/models/granitemoe/__init__.py) ([`granitemoe_test.py`](../../tests/unit/m3/granitemoe_test.py), [`granitemoe_tpu_parity_test.py`](../../tests/unit/m3/granitemoe_tpu_parity_test.py)) | — |
| Alternating Local Sliding-Window / Global Attention + Soft-Cap (`Gemma2`, `Gemma3`) | [`gemma2.py`](../../src/maxtext/models/gemma2.py), [`gemma3.py`](../../src/maxtext/models/gemma3.py) | — |
| Multimodal Vision / Audio Fusion (`Qwen3-VL`, `Qwen3-Omni`) | [`m3/models/qwen3_vl/`](../../src/maxtext/m3/models/qwen3_vl/__init__.py), [`qwen3_omni_layers_test.py`](../../tests/unit/qwen3_omni_layers_test.py) | — |

---

## Topic Index

| Phase | Trigger / Symptom | Stack | Playbook / Recipe | Summary |
| :--- | :--- | :--- | :--- | :--- |
| P4 / Bench | Compile-time or runtime TPU HBM OOM in verification or training/SFT smoke runs (single TPU VM or multi-host cluster) | `linen`, `m3` | [`troubleshooting/hbm_verification.md`](troubleshooting/hbm_verification.md) | 3-stage escalation ladder ("unoptimizing" throughput for minimal chips/HBM: config/sharding flags $\to$ sub-layer remat $\to$ host-RAM expert streaming) |

---

## Directory Domains & Quick Loop

Leaf pages live under four category directories (created as pages are added):
- `troubleshooting/` — TPU HBM, `/dev/shm` storage, XLA compilation, and multi-host infra playbooks.
- `parity/` — Numerical divergence diagnostics (RoPE layouts, MoE router `fp32` tie-flips, norms/multipliers, masks & KV cache).
- `patterns/` — Canonical MaxText Linen & NNX M3 implementation recipes (inhomogeneous `scan`, `ragged_dot` sharding, multimodal fusion).
- `conversion/` — Safetensors $\leftrightarrow$ Orbax parameter hooks, fused projection splits, and strict-`fp32` dtype preservation.

**Operational Loop ("Pre-Flight & Blocker Read $\to$ Distill on Verified Pass"):**
1. **Read**: Check the Archetype table in Phase 1 and the Topic Index above before writing custom workarounds for parity, memory, compilation, or conversion issues.
2. **Distill**: Log raw pitfalls in `{model_name}_handover.md` during bring-up. Once verification passes, follow [`PROTOCOL.md`](PROTOCOL.md) to merge or add reusable lessons and validate via `python3 .agents/wiki/scripts/validate_wiki.py`.
