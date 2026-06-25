import sys

def resolve():
    with open('src/maxtext/layers/quantizations.py', 'r') as f:
        lines = f.readlines()
    
    out = []
    in_conflict = False
    for line in lines:
        if line.startswith('<<<<<<< HEAD'):
            in_conflict = True
            continue
        if line.startswith('>>>>>>> origin/main'):
            in_conflict = False
            continue
        
        if not in_conflict:
            out.append(line)

    with open('src/maxtext/layers/quantizations.py', 'w') as f:
        f.writelines(out)

resolve()
