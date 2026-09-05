import numpy as np, sys
L="/mnt/disks/persist/pr4925_repro"; TAG=sys.argv[1] if len(sys.argv)>1 else "bf16"; SUF=sys.argv[2] if len(sys.argv)>2 else ""
ro=np.load(f"{L}/rollout35b_{TAG}.npz"); mt=np.load(f"{L}/maxtext35b_score_{TAG}{SUF}.npz"); S=512; G=ro["gen_ids"].shape[1]
def band(lt, li, name):
  m=np.isfinite(lt)&np.isfinite(li); d=lt[m]-li[m]; r=np.exp(d)
  print(f"  {name:<40} n={m.sum()} in[0.999,1.002]={np.mean((r>=0.999)&(r<=1.002)):6.2%} ±1%={np.mean((r>=0.99)&(r<=1.01)):5.1%} ±5%={np.mean((r>=0.95)&(r<=1.05)):5.1%} [0.8,1.25]={np.mean((r>=0.8)&(r<=1.25)):5.1%} | |dlogp| med={np.median(np.abs(d)):.4f} mean={np.mean(np.abs(d)):.4f} p99={np.percentile(np.abs(d),99):.3f} max={np.abs(d).max():.2f} | mean ratio={r.mean():.4f}")
gen_lp = ro["gen_logp"].reshape(-1)                       # sampler decode-path logprob of sampled token j (input position S-1+j predicts it)
for mode in ("own","replay"):
  tr = mt[f"{mode}_logp"][:, S-1:S-1+G].reshape(-1)
  band(gen_lp, tr, f"OUTPUT tokens (decode path), {mode} routing")
plp = ro["prompt_logp"][:, 1:].reshape(-1)
for mode in ("own","replay"):
  tr = mt[f"{mode}_logp"][:, :S-1].reshape(-1)
  band(plp, tr, f"PROMPT tokens (prefill path), {mode} routing")
print("  sampled-token stats: sampler mean logp", np.nanmean(gen_lp), "; trainer(own) mean", np.nanmean(mt["own_logp"][:, S-1:S-1+G]))
