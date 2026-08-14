import re

with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/checkpoint_conversion/utils/param_mapping.py', 'r') as f:
    content = f.read()

old_content = """def GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"MaxText↔HF weight-path map for Gemma 4 small (E2B / E4B).

  The small variants thread per-layer state (PLE input + donor K/V) through
  the layer loop, so scanned blocks are not supported.
  \"\"\"
  if scan_layers:
    raise NotImplementedError("Scan layers is not supported for the gemma4_small decoder block.")"""

new_content = """def GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  \"\"\"MaxText↔HF weight-path map for Gemma 4 small (E2B / E4B).\"\"\"
"""

if old_content in content:
    content = content.replace(old_content, new_content)
else:
    print("Old content 1 not found")

old_for = """  for lyr in range(nlayers):
    prefix = f"params-decoder-layers_{lyr}"
    hf_prefix = f"{text_base}.layers.{lyr}\""""

new_for = """  for lyr in range(nlayers):
    prefix = f"params-decoder-scanned_blocks-layers_{lyr}" if scan_layers else f"params-decoder-layers_{lyr}"
    hf_prefix = f"{text_base}.layers.{lyr}\""""

if old_for in content:
    content = content.replace(old_for, new_for)
    with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/checkpoint_conversion/utils/param_mapping.py', 'w') as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("Old for loop not found")

