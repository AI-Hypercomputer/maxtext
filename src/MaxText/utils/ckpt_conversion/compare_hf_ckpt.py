# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Verify the converted safetensor checkpoint (GCS or local) matches the HuggingFace checkpoint reference. 

Usage to compare converted safetensor with remote HF reference:
JAX_PLATFORMS=cpu python3 -m MaxText.utils.ckpt_conversion.compare_hf_ckpt src/maxtext/configs/base.yml \
    model_name=<maxtext_model_name> \
    hf_access_token=<your_hf_token> \
    hardware=cpu \
    --candidate_path=<gcs_bucket_path or local_path> \
    --atol=1e-2 --rtol=1e-2 --max_workers=12

Usage to compare converted safetensor with GCS/Local HF reference:
JAX_PLATFORMS=cpu python3 -m MaxText.utils.ckpt_conversion.compare_hf_ckpt src/maxtext/configs/base.yml \
    hardware=cpu \
    --candidate_path=<gcs_bucket_path or local_path> \
    --reference_path=<gcs_bucket_path or local_path> \
    --atol=1e-2 --rtol=1e-2 --max_workers=12
"""

import argparse
import os
import sys
import numpy as np
import gcsfs
import glob
import jax
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence, Dict

from tqdm import tqdm
import time
from absl import logging
from safetensors.torch import load as load_safetensors
from safetensors import safe_open

from MaxText import pyconfig
from MaxText.utils.ckpt_conversion.utils.utils import HF_IDS, print_ram_usage, get_hf_model
from maxtext.utils import max_logging


jax.config.update("jax_platform_name", "cpu")


def _load_gcs_shard(gcs_path: str, fs: gcsfs.GCSFileSystem) -> Dict[str, np.ndarray]:
  """Worker function to read and process a single safetensors file from GCS."""
  max_logging.log(f"Processing GCS shard: {gcs_path}")

  # Read bytes
  with fs.open(gcs_path, "rb") as f:
    file_bytes = f.read()

  # Parse Safetensors
  loaded_tensors = load_safetensors(file_bytes)

  # Convert to Numpy
  shard_dict = {}
  for key, tensor in loaded_tensors.items():
    shard_dict[key] = tensor.numpy()

  return shard_dict


def _load_local_shard(file_path: str) -> Dict[str, np.ndarray]:
  """Worker function to read and process a single safetensors file from local disk using safe_open."""
  max_logging.log(f"Processing local shard: {file_path}")

  shard_dict = {}
  with safe_open(file_path, framework="pt", device="cpu") as f:
    for key in f.keys():
      loaded_tensors = f.get_tensor(key)
      shard_dict[key] = loaded_tensors.numpy()

  return shard_dict


def load_safetensors_generic(path: str, max_workers: int) -> Dict[str, np.ndarray]:
  """Downloads and loads all .safetensors files from GCS or Local Path in parallel."""

  final_tensor_dict = {}
  futures_map = {}

  is_gcs = path.startswith("gs://")
  print_ram_usage(f"Start {'GCS' if is_gcs else 'Local'} Load")

  with ThreadPoolExecutor(max_workers=max_workers) as executor:

    if is_gcs:
      fs = gcsfs.GCSFileSystem()
      search_pattern = f"{path.rstrip('/')}/*.safetensors"
      safetensor_files = fs.glob(search_pattern)
      safetensor_files = [f"gs://{f}" for f in safetensor_files]

      max_logging.log(f"Found {len(safetensor_files)} files in GCS. Loading...")
      for f in safetensor_files:
        futures_map[executor.submit(_load_gcs_shard, f, fs)] = f

    else:
      # Local filesystem
      search_pattern = os.path.join(path, "*.safetensors")
      safetensor_files = glob.glob(search_pattern)

      max_logging.log(f"Found {len(safetensor_files)} files locally. Loading...")
      for f in safetensor_files:
        futures_map[executor.submit(_load_local_shard, f)] = f

    # Process results
    for future in as_completed(futures_map):
      try:
        shard_data = future.result()
        final_tensor_dict.update(shard_data)
      except Exception as e:
        max_logging.log(f"ERROR: Exception while loading shard: {e}")
        raise e

  print_ram_usage("End Load")
  return final_tensor_dict


def get_hf_model_state_dict(model_id: str, token: str) -> Dict[str, np.ndarray]:
  """Loads the HuggingFace model state dict and converts to numpy."""
  max_logging.log(f"Loading reference model from HuggingFace: {model_id}...")

  hf_model = get_hf_model(model_id, token)
  state_dict = hf_model.state_dict()
  numpy_state_dict = {k: v.numpy() for k, v in state_dict.items()}

  return numpy_state_dict


def verify_dictionaries(
    ref_dict: Dict[str, np.ndarray], cand_dict: Dict[str, np.ndarray], rtol: float, atol: float
) -> bool:
  """Compares two dictionaries of numpy arrays."""

  max_logging.log(f"Verifying with rtol={rtol}, atol={atol}")

  # 1. Compare Keys
  ref_keys = set(ref_dict.keys())
  cand_keys = set(cand_dict.keys())

  if ref_keys != cand_keys:
    max_logging.log("❌ KEYS DO NOT MATCH")
    max_logging.log(f"Missing in Candidate: {ref_keys - cand_keys}")
    max_logging.log(f"Extra in Candidate: {cand_keys - ref_keys}")
    return False

  max_logging.log("✅ Keys match. Verifying tensor values...")

  # 2. Compare Values (Early Return)
  for key in tqdm(ref_keys, desc="Verifying tensors"):
    arr_ref = ref_dict[key]
    arr_cand = cand_dict[key]

    # Check Shape
    if arr_ref.shape != arr_cand.shape:
      max_logging.log(f"❌ SHAPE MISMATCH found for '{key}'")
      max_logging.log(f"Reference: {arr_ref.shape} vs Candidate: {arr_cand.shape}")
      return False
    max_logging.log(f"✅ Key: {key} shape match.")

    # Check Values
    if not np.allclose(arr_ref, arr_cand, rtol=rtol, atol=atol):
      max_diff = np.max(np.abs(arr_ref - arr_cand))
      max_logging.log(f"❌ VALUE MISMATCH found for '{key}'")
      max_logging.log(f"Max difference: {max_diff} (exceeds rtol={rtol}, atol={atol})")
      return False
    max_logging.log(f"✅ Key: {key} value match.")

  max_logging.log("✅ All values match!")
  return True


def main(args: Sequence[str], test_args: argparse.Namespace) -> None:
  # 1. Load Reference (HuggingFace)
  t0 = time.perf_counter()
  if test_args.reference_path:
    hf_state_dict = load_safetensors_generic(test_args.reference_path, test_args.max_workers)
  else:
    config = pyconfig.initialize(args)
    model_name = config.model_name
    if model_name not in HF_IDS:
      raise ValueError(f"Unsupported model name: {model_name}. " f"Supported: {list(HF_IDS.keys())}")

    model_id = HF_IDS[model_name]
    hf_token = config.hf_access_token
    hf_state_dict = get_hf_model_state_dict(model_id, hf_token)
  t1 = time.perf_counter()
  max_logging.log(f"⏱️ HuggingFace model loaded in {(t1 - t0) / 60:.2f} minutes")

  # 2. Load Candidate (GCS or Local)
  t0 = time.perf_counter()
  cand_state_dict = load_safetensors_generic(test_args.candidate_path, test_args.max_workers)
  t1 = time.perf_counter()
  max_logging.log(f"⏱️ Safetensors checkpoint loaded in {(t1 - t0) / 60:.2f} minutes")

  # 3. Compare
  success = verify_dictionaries(hf_state_dict, cand_state_dict, rtol=test_args.rtol, atol=test_args.atol)

  if not success:
    sys.exit(1)


if __name__ == "__main__":
  # Suppress TF logging
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  # Parse script-specific arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--candidate_path",
      type=str,
      default="",
      required=True,
      help="The path to the converted safetensors checkpoint (e.g. gs://bucket/path or /local/path)",
  )

  parser.add_argument(
      "--reference_path",
      type=str,
      default="",
      required=False,
      help="The path to the reference safetensors checkpoint (e.g. gs://bucket/path or /local/path)",
  )

  parser.add_argument(
      "--max_workers",
      type=int,
      default=12,
      required=False,
      help="The max workers for loading safetensors",
  )

  parser.add_argument(
      "--rtol",
      type=float,
      default=1e-2,
      required=False,
      help="Relative tolerance for numpy.allclose",
  )

  parser.add_argument(
      "--atol",
      type=float,
      default=1e-2,
      required=False,
      help="Absolute tolerance for numpy.allclose",
  )

  local_args, _ = parser.parse_known_args()
  logging.set_verbosity(logging.INFO)

  # Filter args for MaxText config parsing
  model_args = sys.argv
  to_remove_args = ["--candidate_path", "--reference_path", "--max_workers", "--rtol", "--atol"]
  model_args = [s for s in model_args if not any(s.startswith(a) for a in to_remove_args)]

  main(model_args, local_args)
