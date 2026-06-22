# Copyright 2023–2026 Google LLC
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
Download a HuggingFace multimodal dataset, unzip video archives,
and generate local Parquet metadata files.
"""

import argparse
import os
import re
import shutil
import tarfile
import zipfile
from huggingface_hub import hf_hub_download, list_repo_files
from datasets import load_dataset
import pyarrow.parquet as pq


def parse_args():
  parser = argparse.ArgumentParser(
      description="Download and prepare local multimodal dataset from HuggingFace Hub."
  )
  parser.add_argument(
      "--repo_id",
      required=True,
      help="HuggingFace dataset repository ID (e.g. lmms-lab/LLaVA-Video-178K)",
  )
  parser.add_argument(
      "--subset",
      required=True,
      help="Subset directory inside repository (e.g. 0_30_s_academic_v0_1)",
  )
  parser.add_argument(
      "--dataset_dir",
      required=True,
      help="Target local directory to write both videos and parquets.",
  )
  parser.add_argument(
      "--split",
      default="all",
      choices=["all", "caption", "open_ended", "multi_choice"],
      help="Specific split to prepare (default: all).",
  )
  parser.add_argument(
      "--token",
      default=None,
      help="HuggingFace access token for gated datasets.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  
  # Ensure the target local directory exists
  os.makedirs(args.dataset_dir, exist_ok=True)
  
  print(f"Connecting to HuggingFace Hub to scan '{args.repo_id}' under subset '{args.subset}'...")
  try:
    all_files = list_repo_files(repo_id=args.repo_id, repo_type="dataset", token=args.token)
  except Exception as e:
    print(f"Error accessing HuggingFace repository: {e}")
    return

  subset_prefix = args.subset.strip("/") + "/"
  target_files = [f for f in all_files if f.startswith(subset_prefix)]
  
  if not target_files:
    print(f"Error: No files found in HuggingFace repo matching subset prefix '{subset_prefix}'")
    return

  # 1. Separate JSON metadata files based on split choice
  split_patterns = {
      "caption": r".*cap.*\.json",
      "open_ended": r".*oe.*\.json",
      "multi_choice": r".*mc.*\.json",
  }
  
  json_files = []
  if args.split == "all":
    json_files = [f for f in target_files if f.endswith(".json")]
  else:
    pattern = split_patterns[args.split]
    json_files = [f for f in target_files if f.endswith(".json") and re.match(pattern, os.path.basename(f))]

  if not json_files:
    print(f"Error: No metadata JSON files found matching split choice '{args.split}'")
    return

  # 2. Identify video archive files
  tar_files = [f for f in target_files if f.endswith(".tar.gz") or f.endswith(".tar") or f.endswith(".zip")]
  
  print("\n" + "="*80)
  print(f"DATASET PREPARATION PLAN")
  print(f"HuggingFace Repo:  {args.repo_id}")
  print(f"Subset / Split:    {args.subset} / {args.split}")
  print(f"Local Directory:   {args.dataset_dir}")
  print(f"Metadata JSONs:    {len(json_files)}")
  print(f"Video Archives:    {len(tar_files)}")
  print("="*80 + "\n")

  # 3. Download and extract video archives
  staging_dir = os.path.join(args.dataset_dir, ".staging")
  os.makedirs(staging_dir, exist_ok=True)
  
  downloaded_archives = []
  for i, f in enumerate(tar_files):
    filename = os.path.basename(f)
    print(f"[{i+1}/{len(tar_files)}] Downloading video archive: {filename} ...")
    try:
      local_path = hf_hub_download(
          repo_id=args.repo_id,
          filename=f,
          repo_type="dataset",
          local_dir=staging_dir,
          token=args.token
      )
      downloaded_archives.append(local_path)
    except Exception as e:
      print(f"Failed to download archive {filename}: {e}")
      return

  for i, archive_path in enumerate(downloaded_archives):
    print(f"[{i+1}/{len(downloaded_archives)}] Extracting video archive: {os.path.basename(archive_path)} ...")
    try:
      if archive_path.endswith(".tar.gz") or archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r:gz" if archive_path.endswith(".tar.gz") else "r:") as tar:
          tar.extractall(path=args.dataset_dir)
      elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
          zip_ref.extractall(path=args.dataset_dir)
    except Exception as e:
      print(f"Failed to extract archive {archive_path}: {e}")
      return

  # Clean up staging directory
  shutil.rmtree(staging_dir)
  print("Video archives extracted successfully. Staging directory cleaned.")

  # 4. Download and convert JSON files to local Parquet files
  local_json_paths = []
  for i, f in enumerate(json_files):
    filename = os.path.basename(f)
    print(f"[{i+1}/{len(json_files)}] Downloading metadata JSON: {filename} ...")
    try:
      local_path = hf_hub_download(
          repo_id=args.repo_id,
          filename=f,
          repo_type="dataset",
          local_dir=args.dataset_dir,
          token=args.token
      )
      local_json_paths.append(local_path)
    except Exception as e:
      print(f"Failed to download metadata JSON {filename}: {e}")
      return

  print("\nConverting JSON files to local Parquet format...")
  try:
    ds = load_dataset("json", data_files=local_json_paths, split="train")
    table = ds.data.table
    
    # Target filename indicates subset/split configurations
    parquet_filename = f"llava-video-178k-{args.split}-00000-of-00001.parquet"
    output_parquet_path = os.path.join(args.dataset_dir, parquet_filename)
    pq.write_table(table, output_parquet_path, compression="zstd")
    
    print(f"Success! Local parquet file generated at: {output_parquet_path}")
  except Exception as e:
    print(f"Error during JSON-to-Parquet conversion: {e}")
    return
  finally:
    # Always clean up temporary JSON files
    for p in local_json_paths:
      if os.path.exists(p):
        os.remove(p)

    # Clean up empty directories created for JSON files
    for p in local_json_paths:
      dirname = os.path.dirname(p)
      target = os.path.abspath(args.dataset_dir)
      while dirname and os.path.abspath(dirname) != target:
        try:
          if not os.listdir(dirname):
            os.rmdir(dirname)
          else:
            break
        except OSError:
          break
        dirname = os.path.dirname(dirname)

  print(f"\nAll operations completed successfully! Dataset is ready locally at: {args.dataset_dir}\n")


if __name__ == "__main__":
  main()
