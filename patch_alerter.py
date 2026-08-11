import os

filepath = "src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/monitor/alerter.py"
with open(filepath, "r") as f:
    content = f.read()

old_metrics_block = '''              "metrics": f"""Diagnosis: {diag}
Hypothesis: {hypothesis}
Config applied: {config}""",'''

new_metrics_block = '''              "metrics": (
                  f"Diagnosis: {diag}\\n"
                  f"Hypothesis: {hypothesis}\\n"
                  f"Config applied: {config}"
              ),'''

if old_metrics_block in content:
    content = content.replace(old_metrics_block, new_metrics_block)
    with open(filepath, "w") as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Block not found!")
