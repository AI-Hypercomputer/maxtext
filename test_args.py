import sys
sys.path.append("src")
from maxtext.configs import pyconfig

argv = ["base_output_directory=gs://test"]
config_path, remaining = pyconfig._resolve_or_infer_config(argv)
print(f"config_path: {config_path}")
print(f"remaining: {remaining}")
