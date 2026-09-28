"""Analytic per-device roofline for olmo35-tiny training on Ironwood (tpu7x).

One step, fwd+bwd, no remat, pdb 3, seq 8192, FSDP 32 x DP 4 (128 devices).
Matmul FLOPs are 3x forward. Bytes are the minimum HBM traffic of a good
kernel: each operand read once, each result written once.
Usage: python scripts/olmo35_roofline.py
"""

PEAK, HBM = 1153.5e12, 3.69e12  # per device (a v7x chip holds two)
B, T, D, V = 3, 8192, 1024, 100352
N = B * T  # tokens per device
L_KDA, L_ATT, L_MOE = 14, 2, 15
E, K, DL, DH = 512, 16, 512, 1024  # experts, top-k, latent (expert input), expert hidden
R = N * K  # routed rows
BF = 2


def mm(m, k, n):  # fwd+bwd FLOPs of an [m,k]x[k,n] matmul
  return 6 * m * k * n


def row(name, flops, byts, note=""):
  t_c, t_h = flops / PEAK * 1e3, byts / HBM * 1e3
  return (name, flops / 1e12, byts / 1e9, t_c, t_h, max(t_c, t_h), "compute" if t_c >= t_h else "HBM", note)


def gemm_bytes(m, k, n, groups=1):
  return BF * (m * k + groups * k * n + m * n)


rows = []
rows.append(row("LM head + loss", mm(N, D, V), 3 * BF * (D * V) + 4 * BF * N * D, "vocab-tiled, logits never hit HBM"))
kda_proj = D * D * 2 + D * 2048 * 2 + D * 256 * 3 + 256 * D + 256 * 2048 + D * 8
rows.append(row("KDA projections (14 layers)", L_KDA * 6 * N * kda_proj, L_KDA * 3 * BF * (kda_proj + N * (D * 4 + 2048 * 3)), ""))
rows.append(row("KDA delta-rule core (14 layers)", L_KDA * 186.0e9, L_KDA * 1.76e9, "kda-vs-gdn-kernels.md roofline"))
att_proj = D * 2048 + 2 * D * 512 + D * D
att_core = 3.5 * 4 * B * T * T * 8 * 128 / 2  # causal, fwd + 2.5x bwd, upper bound (packing skips more)
rows.append(row("attention proj + core (2 layers)", L_ATT * (6 * N * att_proj + att_core), L_ATT * 3 * BF * (att_proj + N * D * 6), "full causal = upper bound"))
rows.append(row("dense layer-0 MLP", mm(N, D, 8192) * 3, 3 * BF * (3 * D * 8192 + N * (D * 2 + 8192 * 3)), ""))
moe_small = D * 512 + D * DL + DL * D + 3 * D * DH  # router, latent down/up, shared expert
rows.append(row("MoE router+latent+shared (15)", L_MOE * 6 * N * moe_small, L_MOE * 3 * BF * (moe_small + N * D * 8), ""))
g = gemm_bytes(R, DL, DH, groups=E)
r = row("MoE routed GEMMs, unfused (15)", L_MOE * 9 * 2 * R * DL * DH, L_MOE * 9 * g, "")
per_gemm = max(2 * R * DL * DH / PEAK, g / HBM) * 1e3  # each GEMM is bounded on its own
rows.append(r[:5] + (L_MOE * 9 * per_gemm, "HBM", f"9 GEMMs x max({2 * R * DL * DH / PEAK * 1e3:.3f} ms MXU, {g / HBM * 1e3:.3f} ms HBM)"))
rows.append(row("MoE SwiGLU passes, unfused (15)", L_MOE * 10 * R * DH, L_MOE * BF * R * DH * (3 + 5), "fwd 2 in 1 out, bwd 3 in 2 out"))
fused_bytes = BF * (2 * R * DL + 3 * E * DL * DH) + BF * (4 * R * DL + 6 * E * DL * DH)
rows.append(row("MoE routed, fused SwiGLU kernel (15)", L_MOE * 9 * 2 * R * DL * DH, L_MOE * fused_bytes, "hidden stays in VMEM; replaces the two rows above"))
rows.append(row("MoE dispatch + combine (15)", L_MOE * 2 * R * DL * 2, L_MOE * 4 * BF * (R * DL + N * DL), "gather + weighted sum, fwd and bwd"))
rows.append(row("MoE top-k + dispatch sort (15)", 0, L_MOE * (4 * N * E + 3 * 4 * R), "one read of fp32 logits"))
rows.append(row("norms, residuals, gates (16)", 16 * 40 * N * D, 16 * 40 * BF * N * D, "~40 [N,D] passes per layer"))
p_dev = 12_496_341_632 / 128
rows.append(row("optimizer (AdamW, fp32 state)", 12 * p_dev, 28 * p_dev, "read p,m,v,g, write p,m,v"))

print("| component | TFLOP | min GB | MXU ms | HBM ms | roofline ms | bound | note |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
  print(f"| {r[0]} | {r[1]:.2f} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.1f} | **{r[5]:.1f}** | {r[6]} | {r[7]} |")
skip_fused = "MoE routed, fused SwiGLU kernel (15)"
skip_unfused = ("MoE routed GEMMs, unfused (15)", "MoE SwiGLU passes, unfused (15)")
tot_u = sum(r[5] for r in rows if r[0] != skip_fused)
tot_f = sum(r[5] for r in rows if r[0] not in skip_unfused)
model_tf = 144.3 * 0.735  # MaxText-counted TFLOP per device per step
for name, t in (("unfused MoE", tot_u), ("fused MoE", tot_f)):
  print(f"roofline step, {name}: {t:.0f} ms -> {model_tf / t * 1e3 / 1153.5 * 100:.0f}% MFU at MaxText's {model_tf:.1f} TFLOP/step")
print(f"20% MFU needs {model_tf / (0.2 * 1153.5) * 1e3:.0f} ms/step")
