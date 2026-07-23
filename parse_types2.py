import re

content = open("src/maxtext/configs/types.py").read()
match = re.search(r'class MaxTextConfig\((.*?)\):', content, re.DOTALL)
bases = [b.strip() for b in match.group(1).split(',')]

for i, base in enumerate(bases):
    if not base or base.startswith('#'): continue
    print(i, base)
