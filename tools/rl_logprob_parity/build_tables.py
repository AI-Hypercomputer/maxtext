"""Assemble the TIS-band tables (35B, 397B) from all result npz files present."""
import numpy as np, os, glob
L = "/mnt/disks/persist/pr4925_repro"; S = 512
tok = np.load(f"{L}/real35b_L3.npz")["tokens"]
def band(lt, li):
  m = np.isfinite(lt) & np.isfinite(li); lt = lt[m]; li = li[m]; d = lt - li; r = np.exp(d)
  return dict(n=len(d), inb=np.mean((r >= 0.999) & (r <= 1.002)), p1=np.mean((r >= 0.99) & (r <= 1.01)), p5=np.mean((r >= 0.95) & (r <= 1.05)),
              med=np.median(np.abs(d)), p99=np.percentile(np.abs(d), 99), mx=np.abs(d).max())
def fmt(b):
  return f"{b['inb']:6.2%} | {b['p1']:5.1%} | {b['p5']:5.1%} | {b['med']:.4f} / {b['p99']:.3f} / {b['mx']:.2f}"
def hidden(a, b):
  D = a.shape[-1]; a = a.reshape(-1, D); b = b.reshape(-1, D)
  c = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
  return f"— | — | — | — | layer-contribution error: relL2 {np.linalg.norm(a-b)/np.linalg.norm(a):.1%}, tokens with contribution cos<0.999: {np.mean(c<0.999):.1%} (no logprob at depth 1);"
def load(p):
  return np.load(p) if os.path.exists(p) else None
rows = []
def row(model, sampler, dtype, replay, depth, text, note=""):
  rows.append((model, sampler, dtype, replay, depth, text, note))
HDR = "| sampler | trainer/sampler dtype | expert replay | layers | in [0.999,1.002] | in ±1% | in ±5% | \\|Δlogp\\| median / p99 / max | note |\n|---|---|---|---|---|---|---|---|---|"

# ---------------- 35B ----------------
mt8 = load(f"{L}/maxtext_prefix.npz"); tx8 = load(f"{L}/torchax_prefix_dp4tp2_ep1.npz")
mt40 = load(f"{L}/maxtext_prefix_nl40_bf16.npz"); tx40 = load(f"{L}/torchax_prefix_dp4tp2_ep1_nl40.npz")
tx8f = load(f"{L}/torchax_prefix_dp4tp2_ep1_nl8_fp8.npz"); mt8f = load(f"{L}/maxtext_prefix_nl8_fp8replay.npz")
tx40f = load(f"{L}/torchax_prefix_dp4tp2_ep1_nl40_fp8.npz"); mt40f = load(f"{L}/maxtext_prefix_nl40_fp8replay.npz")
mtfull = load(f"{L}/maxtext35b_full_logprobs.npz"); vfull = load(f"{L}/vllm35b_prompt_logprobs_dp4.npz"); afull = load(f"{L}/adapter35b_prompt_logprobs_dp4.npz")
mt8q = load(f"{L}/maxtext_prefix_nl40_fp8trainer_full.npz")
z = np.load(f"{L}/real35b_L3.npz")
def lens_rows(model, sampler, dtype, mt, tx, mt_own, prefix_own="own", prefix_rep="replay", depths=(4, 8), extra_full=None):
  for rep, pre, m in (("N", prefix_own, mt_own), ("Y", prefix_rep, mt)):
    if m is None or tx is None: continue
    for k in depths:
      if f"{pre}_lens{k}_logp_actual" in m.files and f"lens{k}_logp_actual" in tx.files:
        note = "full model (true logprobs)" if k >= 40 else "logit lens (probe)"
        row(model, sampler, dtype, rep, str(k) if k < 40 else "40 (full)", fmt(band(m[f"{pre}_lens{k}_logp_actual"], tx[f"lens{k}_logp_actual"])), note)
# native bf16
if mt8 is not None and tx8 is not None:
  row("35B", "tpu-inference native", "bf16 / bf16", "N", "1", hidden(z["train"] - z["h_in"], tx8["y3_on_maxtext_h"] - z["h_in"]), "layer 3 alone, identical input")
  row("35B", "tpu-inference native", "bf16 / bf16", "Y", "1", hidden(mt8["replay_y3"] - z["h_in"], tx8["y3_on_maxtext_h"] - z["h_in"]), "layer 3 alone, identical input")
lens_rows("35B", "tpu-inference native", "bf16 / bf16", mt8, tx8, mt8)
if mt40 is not None and tx40 is not None:
  lens_rows("35B", "tpu-inference native", "bf16 / bf16", mt40, tx40, mt40, depths=(40,))
if mtfull is not None and vfull is not None:
  row("35B", "tpu-inference native (real engine, chunked prefill)", "bf16 / bf16", "N", "40 (full)", fmt(band(mtfull["logp"][:, :-1].reshape(-1), vfull["logp"][:, 1:].reshape(-1))), "true logprobs via prompt_logprobs")
if mt40 is not None and vfull is not None:
  eng = vfull["logp"][:, 1:].reshape(-1)
  row("35B", "tpu-inference native (real engine, chunked prefill)", "bf16 / bf16", "Y", "40 (full)", fmt(band(mt40["replay_lens40_logp_actual"].reshape(8, S)[:, :-1].reshape(-1), eng)),
      "trainer replays the harness-captured torchax routing (engine's own prefill routing not exportable under attn_dp+chunked prefill); trainer own-routing in the same loop: 29.26%")
# real engine with a truncated N-layer model (production scheduler/chunked prefill), vs trainer lens at the same depth
for k in (4, 8):
  e = load(f"{L}/vllm35b_prompt_logprobs_dp4_nl{k}.npz")
  if e is not None and mt8 is not None:
    eng = e["logp"][:, 1:].reshape(-1)
    for rep, pre in (("N", "own"), ("Y", "replay")):
      tr = mt8[f"{pre}_lens{k}_logp_actual"].reshape(8, S)[:, :-1].reshape(-1)
      row("35B", "tpu-inference native (real engine, N-layer model)", "bf16 / bf16", rep, str(k), fmt(band(tr, eng)),
          "engine truncated to N layers via config; trainer = logit lens" + (" (replay routing from harness capture)" if rep == "Y" else ""))
# native fp8 sampler, bf16 trainer
if tx8f is not None:
  own = mt8  # trainer own-routing outputs do not depend on the sampler
  m = {}
  if own is not None:
    for k in own.files: m[k] = own[k]
  if mt8f is not None:
    for k in mt8f.files: m[k] = mt8f[k]
  class D(dict):
    files = property(lambda self: list(self.keys()))
  m = D(m)
  row("35B", "tpu-inference native", "bf16 / fp8", "N", "1", hidden(z["train"] - z["h_in"], tx8f["y3_on_maxtext_h"] - z["h_in"]), "layer 3 alone")
  if "replay_y3" in m: row("35B", "tpu-inference native", "bf16 / fp8", "Y", "1", hidden(m["replay_y3"] - z["h_in"], tx8f["y3_on_maxtext_h"] - z["h_in"]), "layer 3 alone")
  lens_rows("35B", "tpu-inference native", "bf16 / fp8", m, tx8f, m)
if tx40f is not None and mt40f is not None:
  m = {}
  if mt40 is not None:
    for k in mt40.files: m[k] = mt40[k]
  for k in mt40f.files: m[k] = mt40f[k]
  class D(dict):
    files = property(lambda self: list(self.keys()))
  lens_rows("35B", "tpu-inference native", "bf16 / fp8", D(m), tx40f, D(m), depths=(40,))
# fp8 (qwix) trainer + fp8 sampler
if mt8q is not None and tx40f is not None:
  lens_rows("35B", "tpu-inference native", "fp8 (MaxText qwix fp8_full, dynamic) / fp8", mt8q, tx40f, mt8q, depths=(40,))
# MaxText-in-vLLM
if mtfull is not None and afull is not None:
  row("35B", "MaxText in vLLM (attn_dp=4 x tp=2; MoE TP-8, EP not reachable)", "bf16 / bf16", "N", "40 (full)", fmt(band(mtfull["logp"][:, :-1].reshape(-1), afull["logp"][:, 1:].reshape(-1))), "true logprobs via prompt_logprobs")
if afull is not None and vfull is not None:
  row("35B", "calibration: MaxText-in-vLLM engine vs native engine", "bf16 / bf16", "–", "40 (full)", fmt(band(afull["logp"][:, 1:].reshape(-1), vfull["logp"][:, 1:].reshape(-1))), "two samplers vs each other")
mi = load(f"{L}/real35b_L3_dp4tp2.npz")
if mi is not None:
  row("35B", "MaxText in vLLM (layer harness, attn_dp=4 x tp=2, MoE TP-8)", "bf16 / bf16", "N", "1", hidden(z["train"] - z["h_in"], mi["infer"] - z["h_in"]), "layer 3 alone")

# ---------------- 397B ----------------
mo = load(f"{L}/maxtext397_prefix_own.npz"); mr = load(f"{L}/maxtext397_prefix_replay.npz"); t3 = load(f"{L}/torchax397_prefix_dp4tp2_ep1.npz")
if mo is not None and t3 is not None:
  h = mo["own_h_after3"]
  row("397B", "tpu-inference native", "bf16 / fp8", "N", "1", hidden(mo["own_y3"] - h, t3["y3_on_maxtext_h"] - h), "layer 3 alone (sampler FP8: bf16 397B does not fit)")
  if mr is not None: row("397B", "tpu-inference native", "bf16 / fp8", "Y", "1", hidden(mr["replay_y3"] - h, t3["y3_on_maxtext_h"] - h), "layer 3 alone")
  m = {}
  for k in mo.files: m[k] = mo[k]
  if mr is not None:
    for k in mr.files: m[k] = mr[k]
  class D(dict):
    files = property(lambda self: list(self.keys()))
  lens_rows("397B", "tpu-inference native", "bf16 / fp8", D(m), t3, D(m))
qo = load(f"{L}/maxtext397_prefix_own_fp8trainer.npz"); qr = load(f"{L}/maxtext397_prefix_replay_fp8trainer.npz")
if qo is not None and t3 is not None:
  m = {}
  for k in qo.files: m[k] = qo[k]
  if qr is not None:
    for k in qr.files: m[k] = qr[k]
  class D(dict):
    files = property(lambda self: list(self.keys()))
  lens_rows("397B", "tpu-inference native", "fp8 (MaxText qwix fp8_full, dynamic) / fp8", D(m), t3, D(m), depths=(8,))

out = []
for model in ("35B", "397B"):
  out.append(f"\n### {model}\n{HDR}")
  for r in rows:
    if r[0] == model: out.append(f"| {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
  if model == "397B":
    out.append("| tpu-inference native | bf16 / bf16 | – | – | n/a | n/a | n/a | n/a | bf16 397B (794 GB) exceeds 8x v7x HBM (758 GB) |")
    out.append("| any | any | any | 60 (full) | n/a | n/a | n/a | n/a | bf16 trainer cannot hold 60 layers on this host; prefix of 8 only |")
    out.append("| MaxText in vLLM | any | – | – | n/a | n/a | n/a | n/a | adapter cannot load 397B (bf16 only, 794 GB) nor FP8 checkpoints |")
  out.append("| MaxText in vLLM | bf16 / fp8, fp8 / fp8 | – | – | n/a | n/a | n/a | n/a | adapter has no FP8 checkpoint path |" if model == "35B" else "")
print("\n".join(o for o in out if o is not None))
