import numpy as np, sys
L = "/mnt/disks/persist/pr4925_repro"; S = 512; import os; D = int(os.environ.get("D", "2048"))
tokens = np.load(f"{L}/real35b_L3.npz")["tokens"].reshape(-1)
tx = np.load(sys.argv[1] if len(sys.argv) > 1 else f"{L}/torchax_prefix_dp4tp2_ep1.npz"); mt = np.load(sys.argv[2] if len(sys.argv) > 2 else f"{L}/maxtext_prefix.npz")
mt_own = np.load(sys.argv[3]) if len(sys.argv) > 3 else mt   # 397B: own-mode results live in a separate npz
h_in = mt_own["own_h_after3"] if "own_h_after3" in mt_own else np.load(f"{L}/real35b_L3.npz")["h_in"]
class _M(dict):
  pass
_mt = {}; 
for k in mt.files: _mt[k] = mt[k]
for k in mt_own.files: _mt.setdefault(k, mt_own[k])
mt = _mt
z = {"train": mt["own_y3"] if "own_y3" in mt else np.load(f"{L}/real35b_L3.npz")["train"]}
def hid(name, a, b, base=None):
  a = a.reshape(-1, D); b = b.reshape(-1, D)
  if base is not None: a = a - base.reshape(-1, D); b = b - base.reshape(-1, D)
  c = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
  print(f"  {name:<50} relL2={np.linalg.norm(a-b)/np.linalg.norm(a):.4e}  per-token cos mean={c.mean():.6f} min={c.min():.4f}  tokens cos<0.999: {np.mean(c<0.999):5.2%}")
def lens(name, k, mode):
  lt = mt[f"{mode}_lens{k}_logp_actual"]; li = tx[f"lens{k}_logp_actual"]
  t1 = np.mean(mt[f"{mode}_lens{k}_top1"] == tx[f"lens{k}_top1"])
  d = lt - li; r = np.exp(d)
  Pt = mt[f"{mode}_lens{k}_logprobs_2seq"]; Pi = tx[f"lens{k}_logprobs_2seq"]
  kl = np.sum(np.exp(Pt) * (Pt - Pi), -1)  # KL(train || sampler) per token, first 2 sequences
  print(f"  {name:<50} |dlogp(actual)| mean={np.mean(np.abs(d)):.4f} p99={np.percentile(np.abs(d),99):.4f} max={np.abs(d).max():.3f} | "
        f"top-1 agree={t1:.2%} | KL(train||sampler) mean={kl.mean():.2e} max={kl.max():.2e} | ratio exp(dlogp): frac outside [0.8,1.25]={np.mean((r<0.8)|(r>1.25)):.2%}")
print("=== layer 3 alone on identical h_in (delta = layer contribution) ===")
hid("train(own routing) vs torchax(bf16 router)", z["train"], tx["y3_on_maxtext_h"], h_in)
hid("train(REPLAY torchax routing) vs torchax", mt["replay_y3"], tx["y3_on_maxtext_h"], h_in)
hid("train(own) vs train(replay)  [routing effect alone]", z["train"], mt["replay_y3"], h_in)
print("=== prefix from real tokens: hidden state after N layers, train vs torchax ===")
for k in (3, 4, 8):
  hid(f"after {k} layers: train(own) vs torchax", mt[f"own_h_after{k}"], tx[f"h_after{k}"])
  hid(f"after {k} layers: train(REPLAY) vs torchax", mt[f"replay_h_after{k}"], tx[f"h_after{k}"])
print("=== logit-lens next-token logprobs (final norm + lm_head on the hidden state), train vs torchax ===")
for k in (4, 8):
  lens(f"after {k} layers: own routing", k, "own")
  lens(f"after {k} layers: REPLAY routing", k, "replay")
print("(sanity) train own vs replay lens8 |dlogp| mean =", np.mean(np.abs(mt["own_lens8_logp_actual"] - mt["replay_lens8_logp_actual"])))
