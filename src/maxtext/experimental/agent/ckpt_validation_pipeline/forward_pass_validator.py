# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Automated Forward Pass Logit Validation Wrapper for MaxText.
This script wraps tests/utils/forward_pass_logit_checker.py to standardise
reporting for the Airflow fail-fast pipeline.
"""

import argparse
import io
import inspect
import json
import os
import re
import runpy
import subprocess
import sys
import traceback
import absl.logging
import maxtext
from maxtext.utils import gcs_utils
from maxtext.utils import model_creation_utils
# pylint: disable=no-name-in-module
from maxtext.utils import max_logging as logger

# Initialize logging verbosity to INFO so logger.info is actually printed
absl.logging.set_verbosity(absl.logging.INFO)


def validate_forward_pass(run_name, internal_model_name, checkpoint_path, report_gcs_dir, unknown_args):
  """Run logit checker as a subprocess and generate a standardized JSON report."""
  logger.info(f"Running Forward Pass Logit Verification for {run_name}...")

  # base command
  command = [
      "python3",
      "tests/utils/forward_pass_logit_checker.py",
      "src/maxtext/configs/base.yml",
      f"model_name={internal_model_name}",
      f"load_parameters_path={checkpoint_path}",
      "dtype=float32",
      "activations_in_float32=true",
      "matmul_precision=high",
      "override_model_config=True",
      "--max_kl_div=0.1",
  ]

  # append additional maxtext configs from unknown args
  if unknown_args:
    logger.info("Applying additional flags from MaxText overrides...")
    for arg in unknown_args:
      command.append(arg)
      logger.info(f"  -> {arg}")

  # find the absolute path to the root of the repository
  maxtext_module_dir = os.path.dirname(maxtext.__file__)
  repo_root = os.path.abspath(os.path.join(maxtext_module_dir, "../../"))

  # applying a monkeypatch to maxtext's model_creation_utils because it has a bug where
  # it cannot resolve SequenceKey (list indices) to string keys in Linen checkpoints.

  source = inspect.getsource(model_creation_utils._fix_restore_args_for_shape_mismatch)  # pylint: disable=protected-access

  new_lookup = """  def _lookup_stored_meta(path):
    # Monkeypatched to handle NNX to Linen structural mismatches
    def _navigate(p):
      node = stored_metadata_tree
      for key in p:
        if isinstance(key, jax.tree_util.SequenceKey):
          if isinstance(node, (list, tuple)) and 0 <= key.idx < len(node):
            node = node[key.idx]
            continue
          if isinstance(node, dict) and str(key.idx) in node:
            node = node[str(key.idx)]
            continue
          return None
        if isinstance(node, (list, tuple)):
          name = _key_str(key)
          if name.isdigit() and 0 <= int(name) < len(node):
            node = node[int(name)]
            continue
          return None
        if not isinstance(node, dict):
          return None
        name = _key_str(key)
        if name in node:
          node = node[name]
          continue
        raw = str(key)
        if raw in node:
          node = node[raw]
          continue
        if name == "pre_self_attention_layer_norm" and "input_layernorm" in node:
          node = node["input_layernorm"]
          continue
        if name == "post_self_attention_layer_norm" and "post_attention_layernorm" in node:
          node = node["post_attention_layernorm"]
          continue
        if name == "self_attention" and "attention" in node:
          node = node["attention"]
          continue
        if name == "input_layernorm" and "pre_self_attention_layer_norm" in node:
          node = node["pre_self_attention_layer_norm"]
          continue
        if name == "post_attention_layernorm" and "post_self_attention_layer_norm" in node:
          node = node["post_self_attention_layer_norm"]
          continue
        if name == "attention" and "self_attention" in node:
          node = node["self_attention"]
          continue
        return None
      return node

    # Try navigating the original path first (for Linen-Linen or NNX-NNX)
    res = _navigate(path)
    if res is not None:
      return res

    # Otherwise fallback to converting layers.0 -> layers_0 and navigate
    new_path = []
    i = 0
    while i < len(path):
      k_str = _key_str(path[i])
      if i + 1 < len(path) and k_str.endswith("layers"):
        next_k_str = _key_str(path[i+1])
        if next_k_str.isdigit():
          new_path.append(f"{k_str}_{next_k_str}")
          i += 2
          continue
      new_path.append(path[i])
      i += 1

    return _navigate(new_path)"""

  target_lookup = r"  def _lookup_stored_meta\(path\):[\s\S]*?(?=\n\s*mismatched_paths_sharded = \[\])"
  patched_source = re.sub(target_lookup, new_lookup, source)
  if patched_source == source:
    raise RuntimeError(
        "Failed to apply the monkeypatch to _fix_restore_args_for_shape_mismatch. "
        "The target regex pattern was not found in model_creation_utils.py."
    )

  env = dict(model_creation_utils.__dict__)
  exec(patched_source, env)  # pylint: disable=exec-used

  _original_fix_restore = model_creation_utils._fix_restore_args_for_shape_mismatch  # pylint: disable=protected-access
  model_creation_utils._fix_restore_args_for_shape_mismatch = env[  # pylint: disable=protected-access
      "_fix_restore_args_for_shape_mismatch"
  ]

  import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel

  _original_restore = ocp.Checkpointer.restore

  def _monkeypatched_restore(self, directory, item=None, transforms=None, restore_args=None, **kwargs):
    def _rename_nnx_linen_keys(tree, to_linen: bool):
      """Recursively map parameter key names and layer hierarchy between NNX and Linen conventions.

      Linen checkpoints on disk (e.g. Qwen3-8B unscanned) store weights under:
        - decoder/layers/0/input_layernorm
        - decoder/layers/0/post_attention_layernorm
        - decoder/layers/0/attention

      NNX Qwen3DecoderLayer (inheriting from AttentionWithNorm) expects:
        - decoder/layers_0/pre_self_attention_layer_norm
        - decoder/layers_0/post_self_attention_layer_norm
        - decoder/layers_0/self_attention

      When `to_linen=True` (before Orbax restore):
        - Converts NNX layer attributes (`layers_0`, `layers_1`) into Linen sequence
          dictionary `layers: {'0': ..., '1': ...}`
        - Maps NNX normalization/attention attribute names to Linen checkpoint key names.
      When `to_linen=False` (after Orbax restore):
        - Unpacks Linen `layers: {'0': ..., '1': ...}` sequence dictionary back into direct
          NNX attributes (`layers_0`, `layers_1`).
        - Maps Linen checkpoint key names back to NNX attribute names so
          nnx.update(model, checkpoint) populates all weights.
      """
      if to_linen:
        key_map = {
            "pre_self_attention_layer_norm": "input_layernorm",
            "post_self_attention_layer_norm": "post_attention_layernorm",
            "self_attention": "attention",
        }
      else:
        key_map = {
            "input_layernorm": "pre_self_attention_layer_norm",
            "post_attention_layernorm": "post_self_attention_layer_norm",
            "attention": "self_attention",
        }

      # Recursively traverse dictionaries or dictionary-like mappings (including nnx.State)
      if isinstance(tree, dict) or hasattr(tree, "items"):
        new_tree = {}
        for k, v in tree.items():
          k_str = str(k)
          # Replace key if it matches our mapping; otherwise keep original key name
          new_k = key_map.get(k_str, k)
          new_tree[new_k] = _rename_nnx_linen_keys(v, to_linen=to_linen)

        if to_linen:
          # Convert NNX layers_0, layers_1 -> Linen sequence dict layers: {'0': ..., '1': ...}
          layer_keys = [k for k in list(new_tree.keys()) if re.match(r"^layers_(\d+)$", str(k))]
          if layer_keys:
            layers_dict = {}
            for lk in layer_keys:
              idx_str = re.match(r"^layers_(\d+)$", str(lk)).group(1)
              layers_dict[idx_str] = new_tree.pop(lk)
            new_tree["layers"] = layers_dict
        else:
          # Convert Linen sequence dict layers: {'0': ..., '1': ...} -> NNX layers_0, layers_1
          if "layers" in new_tree and (isinstance(new_tree["layers"], dict) or hasattr(new_tree["layers"], "items")):
            layers_dict = new_tree.pop("layers")
            for idx_key, layer_val in layers_dict.items():
              new_tree[f"layers_{idx_key}"] = layer_val

        try:
          return type(tree)(new_tree)
        except Exception:  # pylint: disable=broad-exception-caught
          return new_tree

      # Recursively traverse lists or tuples (e.g. sequences of layers or restore args)
      if isinstance(tree, (list, tuple)):
        return type(tree)(_rename_nnx_linen_keys(x, to_linen=to_linen) for x in tree)

      # Return leaf arrays / primitives unmodified
      return tree

    # When restoring an NNX model from a Linen checkpoint without explicit transforms,
    # detect whether the checkpoint actually uses Linen conventions before translating keys.
    # If the checkpoint is already in NNX format (e.g. qwen3-8b unscanned), pass item unchanged.
    if item is not None and restore_args is not None and not transforms:
      is_linen_ckpt = False
      try:
        meta = self.metadata(directory)
        item_meta = meta.item_metadata if hasattr(meta, "item_metadata") and meta.item_metadata is not None else meta
        if item_meta is not None:
          flat_meta = ocp.tree.to_flat_dict(item_meta)
          meta_keys_str = " ".join(".".join(map(str, k)) for k in flat_meta.keys())
          if "input_layernorm" in meta_keys_str or ".layers.0." in meta_keys_str or ".layers.1." in meta_keys_str:
            is_linen_ckpt = True
      except Exception as e:  # pylint: disable=broad-exception-caught
        absl.logging.info("Could not inspect checkpoint metadata for Linen conventions: %s", e)

      if is_linen_ckpt:
        linen_item = _rename_nnx_linen_keys(item, to_linen=True)
        linen_restore_args = _rename_nnx_linen_keys(restore_args, to_linen=True)
        restored_linen = _original_restore(
            self,
            directory,
            item=linen_item,
            transforms=transforms,
            restore_args=linen_restore_args,
            **kwargs,
        )
        return _rename_nnx_linen_keys(restored_linen, to_linen=False)

    return _original_restore(
        self,
        directory,
        item=item,
        transforms=transforms,
        restore_args=restore_args,
        **kwargs,
    )

  ocp.Checkpointer.restore = _monkeypatched_restore

  import jax  # pylint: disable=import-outside-toplevel

  import transformers  # pylint: disable=import-outside-toplevel

  _orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

  def _monkeypatched_from_pretrained(*p_args, **p_kwargs):
    tokenizer = _orig_from_pretrained(*p_args, **p_kwargs)
    if getattr(tokenizer, "pad_token", None) is None:
      if getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
      elif getattr(tokenizer, "unk_token", None) is not None:
        tokenizer.pad_token = tokenizer.unk_token
      else:
        try:
          tokenizer.add_special_tokens({"pad_token": "<pad>"})
        except Exception:
          tokenizer.pad_token_id = 0
    return tokenizer

  transformers.AutoTokenizer.from_pretrained = _monkeypatched_from_pretrained

  # run script in same process to apply monkeypatch
  old_stdout = sys.stdout
  old_stderr = sys.stderr
  sys.stdout = stdout_cap = io.StringIO()
  sys.stderr = stderr_cap = io.StringIO()

  import logging  # pylint: disable=import-outside-toplevel

  handlers_to_restore = []
  for logger_name in [None, "absl"]:
    l = logging.getLogger(logger_name)
    for h in l.handlers:
      if isinstance(h, logging.StreamHandler):
        old_stream = h.stream
        if old_stream is old_stderr:
          h.setStream(sys.stderr)
          handlers_to_restore.append((h, old_stream))
        elif old_stream is old_stdout:
          h.setStream(sys.stdout)
          handlers_to_restore.append((h, old_stream))

  old_cwd = os.getcwd()
  os.chdir(repo_root)

  returncode = 0
  try:
    sys.argv = command[1:]
    runpy.run_path("tests/utils/forward_pass_logit_checker.py", run_name="__main__")
  except SystemExit as e:
    returncode = e.code if e.code is not None else 0
  except Exception:  # pylint: disable=broad-exception-caught
    traceback.print_exc(file=sys.stderr)
    returncode = 1
  finally:
    ocp.Checkpointer.restore = _original_restore
    model_creation_utils._fix_restore_args_for_shape_mismatch = _original_fix_restore  # pylint: disable=protected-access
    if _orig_array_delete is not None:
      jax.Array.delete = _orig_array_delete
    transformers.AutoTokenizer.from_pretrained = _orig_from_pretrained

    # Restore logging handlers
    for h, old_stream in handlers_to_restore:
      h.setStream(old_stream)

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    os.chdir(old_cwd)

  stdout_str = stdout_cap.getvalue()
  stderr_str = stderr_cap.getvalue()

  # generate report
  report = {
      "run_name": run_name,
      "model": internal_model_name,
      "status": "SUCCESS" if returncode == 0 else "FAILED",
      "success": returncode == 0,
      "stderr": (stderr_str if returncode != 0 else "Success"),
      "stdout": (stdout_str if returncode != 0 else "Success"),
      "checkpoint_used": checkpoint_path,
      "stage": "forward_pass_validation",
  }

  # build and save report
  report_dir = os.path.join(old_cwd, "reports")
  os.makedirs(report_dir, exist_ok=True)
  output_path = os.path.join(report_dir, f"report_{run_name}_forward_pass.json")

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)
  logger.info(f"Report saved locally to {output_path}")

  # upload to GCS using standard MaxText utils
  if report_gcs_dir:
    try:
      gcs_dir = report_gcs_dir
      if not gcs_dir.endswith("/"):
        gcs_dir += "/"
      gcs_utils.upload_blob(f"{gcs_dir}report_{run_name}_forward_pass.json", output_path)
    except Exception as e:
      logger.error(f"Failed to upload forward pass report to GCS: {e}")

  if returncode != 0:
    logger.info(f"Command STDOUT:\n{stdout_str}")
    logger.error(f"Command STDERR:\n{stderr_str}")
    raise ValueError("ERROR: Forward pass logit verification failed! See logs for details.")

  logger.info("Forward pass validation successful!")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Validate Forward Pass Logits")
  parser.add_argument("--run_name", type=str, required=True, help="Validation run name")
  parser.add_argument(
      "--maxtext_model_name",
      type=str,
      required=True,
      help="Internal MaxText model name",
  )
  parser.add_argument("--checkpoint_gcs_path", type=str, required=True, help="GCS path to checkpoint")
  parser.add_argument("--report_gcs_dir", type=str, default="", help="GCS directory for reports")

  args, unknown = parser.parse_known_args()

  try:
    validate_forward_pass(
        args.run_name,
        args.maxtext_model_name,
        args.checkpoint_gcs_path,
        args.report_gcs_dir,
        unknown,
    )
  except (ValueError, KeyError, subprocess.CalledProcessError) as e:
    logger.error(f"FAILED: {e}")
    # Always fail hard to halt the Airflow DAG
    sys.exit(1)
