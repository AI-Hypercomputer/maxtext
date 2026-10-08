"""Component breakdown of one olmoe3-3p5b v4 train step from an xplane plus scripts/olmoe3_v4_srcsurvey.py records.

  PYTHONPATH=<xla-shell> python scripts/olmoe3_v4_components.py X.xplane.pb Xrecs.pkl ["component|component"]

The optional third argument names components to drill into (top kernels and source lines)."""
import xla_shell, pickle, collections, re, sys
out, _ = pickle.load(open(sys.argv[2], 'rb'))
def strip(n): return re.sub(r'(\.\d+)+(\..*)?$', '', n)
def norm(t): return (t or '').rstrip(':')
acc = collections.defaultdict(collections.Counter); ex = {}
for o in out:
  for key in ((strip(o['name']), norm(o['tf'])), strip(o['name'])):
    acc[key][o['src']] += o['t']; ex.setdefault(key, o['expr'])
src = {k: v.most_common(1)[0][0] for k, v in acc.items()}
def look(d, r, dflt):
  k = (r.kernel, norm(r.tf_op))
  return d.get(k, d.get(strip(r.kernel), dflt))
rows = xla_shell.open_capture(sys.argv[1]).scope(module='jit_train_step').ops(group='kernel_tf_op', enrich=True)
def comp(r):
  k, c, t = r.kernel, r.category, (r.tf_op or '')
  s = look(src, r, '?'); f = s.split(':')[0]; e = look(ex, r, '')
  if 'ragged-all-to-all' in k or 'ragged_all_to_all' in t: return 'MoE ragged all-to-all'
  if c in ('async-start', 'async-done') or k.startswith(('all-gather', 'all-reduce', 'reduce-scatter', 'collective-permute', 'all-to-all')): return 'collectives'
  if 'megablox' in f or k.startswith(('gmm', 'tgmm')) or 'gmm' in t.split('/')[-1]: return 'MoE expert GEMMs (megablox)'
  if 'splash' in k or 'splash' in f: return 'full attention (splash)'
  if 'vocabulary_tiling' in f or '100352' in e: return 'LM head + loss'
  if 'topk' in f or c == 'sort': return 'MoE top-k + sorts'
  if 'moe.py' in f and (c in ('gather', 'scatter') or 'gather' in k or 'scatter' in k): return 'MoE routing, gather, scatter'
  if 'moe.py' in f: return 'MoE other (router, EMo, combine, masks)'
  if 'olmoe3.py' in f and c == 'convolution fusion': return 'KDA matmuls (intra-chunk, inverse, scan)'
  if 'olmoe3.py' in f: return 'KDA elementwise, norms, conv, gates'
  if c == 'convolution fusion': return 'dense matmuls (projections)'
  if c == 'data formatting' or k.startswith(('copy', 'transpose')): return 'copies / relayout'
  if 'normalizations' in f: return 'norms, residuals'
  return 'other (' + c + ')'
B = collections.Counter(); RM = collections.Counter(); S = collections.defaultdict(collections.Counter)
for r in rows:
  n = comp(r); B[n] += r.total_us
  S[n][f"{r.kernel[:40]} [{r.category}] {look(src, r, '?')}"] += r.total_us
  if 'rematted_computation' in (r.tf_op or ''): RM[n] += r.total_us
tot = sum(B.values())
print("| component | ms | share | of which remat ms |"); print("|---|---|---|---|")
for n, v in B.most_common(): print(f"| {n} | {v/1e3:.1f} | {100*v/tot:.1f}% | {RM[n]/1e3:.1f} |")
print(f"| total | {tot/1e3:.1f} | | {sum(RM.values())/1e3:.1f} |")
for n in (sys.argv[3].split('|') if len(sys.argv) > 3 else []):
  print('==', n); [print(f"   {v/1e3:7.1f}  {k}") for k, v in S[n].most_common(12)]
