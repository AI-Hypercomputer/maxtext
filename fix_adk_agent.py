import re

filepath = "src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/adk_agent.py"
with open(filepath, "r") as f:
    text = f.read()

# Fix re-imports inside functions that had different indents
text = re.sub(r"^[ \t]*import urllib\.request\n", "", text, flags=re.MULTILINE)
text = re.sub(r"^[ \t]*import google\.cloud\.storage\n", "", text, flags=re.MULTILINE)
text = re.sub(r"^[ \t]*from google\.cloud import storage\n", "", text, flags=re.MULTILINE)
text = re.sub(r"^[ \t]*import google\.cloud\.storage as gcs\n", "", text, flags=re.MULTILINE)
text = re.sub(r"^[ \t]*import os\n", "", text, flags=re.MULTILINE)

# The top of the file has the imports I placed there. Let's deduplicate them.
text = text.replace("import os\nimport urllib.request\nfrom google.cloud import storage\nfrom monitor.state_manager import get_run_state, record_attempt\nimport subprocess\n", "")
# Re-add at the very top:
top_imports = """
import os
import subprocess
import urllib.request
from google.cloud import storage
"""
text = text.replace('"""\n\nimport json', '"""\n\nimport json' + top_imports)

with open(filepath, "w") as f:
    f.write(text)

