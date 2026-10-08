"""Dump per-op records (name, category, time, source line) of an xplane with xla-shell, for components.

  PYTHONPATH=<xla-shell> python scripts/olmoe3_v4_srcsurvey.py X.xplane.pb Xrecs.pkl

Prints the step time share per source file and writes the records for scripts/olmoe3_v4_components.py.
"""
import collections
import pickle
import re
import sys

import xla_shell

c = xla_shell.open_capture(sys.argv[1])
recs, summ = c.records()
out = []
for r in recs:
  src = re.findall(r"'>([^<]*)</div>", r.source_info or "")
  s = re.sub(r".*/(src/maxtext/|site-packages/)", "", src[0]) if src else "?"
  out.append(
      dict(
          name=r.name,
          cat=r.category,
          t=r.time_us,
          n=r.occurrences,
          src=s,
          expr=(r.expression or "")[:400],
          tf=r.tf_op_name or "",
          bound=r.bound_by,
      )
  )
pickle.dump((out, summ.total_time_us), open(sys.argv[2], "wb"))
g = collections.Counter()
for o in out:
  g[o["src"].split(":")[0]] += o["t"]
tot = sum(g.values())
for k, v in g.most_common(30):
  print(f"{100 * v / tot:5.1f}%  {k}")
