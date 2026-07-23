import re

content = open('src/maxtext/configs/types.py').read()
match = re.search(r'class MaxTextConfig\((.*?)\):', content, re.DOTALL)
if match:
    bases = [b.strip() for b in match.group(1).split(',')]
    print("Bases:")
    for b in bases:
        if b and not b.startswith('#'):
            print(b)
else:
    print("Not found")

