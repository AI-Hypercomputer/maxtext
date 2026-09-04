"""Script to inspect and verify Qwix quantization interception via eval_shape."""

import os
import sys

# Ensure current working directory and src/ are on PYTHONPATH
cwd = os.getcwd()
if cwd not in sys.path:
  sys.path.insert(0, cwd)
src_dir = os.path.join(cwd, "src")
if os.path.isdir(src_dir) and src_dir not in sys.path:
  sys.path.insert(0, src_dir)

import logging
from absl import app
from absl import logging as absl_logging
from flax import nnx
import jax
import jax.numpy as jnp
import qwix

from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.utils import model_creation_utils


def _patch_pyconfig_for_unknown_keys():
  """Filter out unknown keys from YAML to prevent strict validation errors on stale branches."""
  orig_prepare = pyconfig._prepare_for_pydantic

  def safe_prepare(raw_keys, config_class=pyconfig.types.MaxTextConfig):
    valid_fields = set(config_class.model_fields.keys())
    multimodal_fields = getattr(getattr(pyconfig.types, "Multimodal", None), "model_fields", {})
    cleaned_keys = {}
    for k, v in raw_keys.items():
      if k in valid_fields or k in multimodal_fields:
        cleaned_keys[k] = v
      else:
        logging.warning("Ignoring unrecognized config key: %s", k)
    return orig_prepare(cleaned_keys, config_class=config_class)

  pyconfig._prepare_for_pydantic = safe_prepare


def main(argv):
  _patch_pyconfig_for_unknown_keys()

  # 1. Enable DEBUG logging to surface Qwix interception trace logs
  logging.basicConfig(level=logging.DEBUG)
  absl_logging.set_verbosity(absl_logging.DEBUG)
  logging.getLogger("qwix").setLevel(logging.DEBUG)

  # 2. Build configuration with defaults if not provided in CLI
  default_args = [
      "",
      "src/maxtext/configs/base.yml",
      "model_name=deepseek3-671b",
      "quantization=fp8_full",
      "use_qwix_quantization=true",
      "scan_layers=true",
      # "override_model_config=true",
      # "base_num_decoder_layers=4",
      "per_device_batch_size=1",
      "max_target_length=128",
  ]
  config_args = argv if len(argv) > 1 else default_args
  print(f"[INFO] Initializing config with: {config_args}")
  config = pyconfig.initialize(config_args)

  # 3. Create abstract model using nnx.eval_shape (0 FLOPs, 0 device allocation)
  print("\n[INFO] Running create_nnx_abstract_model via eval_shape...")
  _, abstract_model = model_creation_utils.create_nnx_abstract_model(config)
  print("[INFO] Abstract model constructed successfully.\n")

  # 4. Inspect active Qwix rules
  provider = quantizations.get_qt_provider(config)
  if provider:
    print("=" * 80)
    print("ACTIVE QWIX RULES:")
    for idx, rule in enumerate(provider._rules):
      print(f"  Rule [{idx}]: module_path={rule.module_path!r}, op_names={rule.op_names}")
      print(f"            weight_qtype={rule.weight_qtype}, act_qtype={rule.act_qtype}")
    print("=" * 80)

  # 5. Inspect intercepted layers and parameters in the model graph
  print("\nINSPECTING GATE & ROUTED MOE PARAMETERS IN ABSTRACT MODEL:")
  print("-" * 80)
  found_gate = False
  for path, var in nnx.graph.iter_graph(abstract_model):
    path_str = "/".join(map(str, path))
    if any(k in path_str.lower() for k in ("gate", "moe", "quant", "scale")):
      val = getattr(var, "value", var)
      shape_dtype = getattr(val, "shape", None), getattr(val, "dtype", None)
      sharding_info = getattr(val, "sharding", None)
      print(f"Path: {path_str}")
      print(f"  Type: {type(var).__name__}")
      print(f"  Shape/Dtype: shape={shape_dtype[0]}, dtype={shape_dtype[1]}")
      if sharding_info:
        print(f"  Sharding: {sharding_info}")
      print("-" * 80)
      if "gate" in path_str.lower():
        found_gate = True

  if not found_gate:
    print("[WARNING] No 'gate' parameters found. Full graph paths:")
    for path, var in nnx.graph.iter_graph(abstract_model):
      print(" ", "/".join(map(str, path)), "->", type(var).__name__)


if __name__ == "__main__":
  app.run(main)
