import re

content = open("src/maxtext/configs/types.py").read()

# Find all base classes of MaxTextConfig
match = re.search(r'class MaxTextConfig\((.*?)\):', content, re.DOTALL)
bases = [b.strip() for b in match.group(1).split(',')]

all_fields = []

for base in bases:
    if not base or base.startswith('#'): continue
    # find class definition
    class_match = re.search(f'class {base}\([^)]*\):(.*?)(?=\nclass |\Z)', content, re.DOTALL)
    if not class_match: continue
    
    # find fields: "  field_name: type"
    fields = re.findall(r'^\s+([a-zA-Z_0-9]+)\s*:', class_match.group(1), re.MULTILINE)
    all_fields.extend(fields)

print(f"max_position_embeddings in bases? {'max_position_embeddings' in all_fields}")
print(f"original_max_position_embeddings in bases? {'original_max_position_embeddings' in all_fields}")
