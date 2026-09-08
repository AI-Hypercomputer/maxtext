# Copyright 2026 Google LLC
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

"""Script to download and prepare LLaVA-Video-178K dataset for MaxText.

Usage example:
  python3 tools/data_generation/prepare_llava_video_178k.py \
    --output_dir /your_data_path/LLaVA-Video-178K \
    --fold 0_30_s_academic_v0_1 \
    --download_videos
"""

import argparse
import os
import shutil
import subprocess
import tarfile

import datasets
from huggingface_hub import hf_hub_download


def main():
  parser = argparse.ArgumentParser(description="Prepare LLaVA-Video-178K dataset for MaxText.")
  parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the prepared dataset.")
  parser.add_argument("--fold", type=str, default="0_30_s_academic_v0_1", help="Fold to download.")
  parser.add_argument("--download_videos", action="store_true", help="Whether to download videos (large).")
  parser.add_argument(
      "--video_subset",
      type=int,
      default=None,
      help="Number of video tars to download (1-8). If None, downloads all.",
  )
  parser.add_argument(
      "--local_temp_dir",
      type=str,
      default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "llava_video_178k_temp"),
      help="Local directory for temporary storage when uploading to GCS.",
  )
  args = parser.parse_args()

  is_gcs = args.output_dir.startswith("gs://")
  if is_gcs:
    local_output_dir = args.local_temp_dir
    print(f"GCS output directory detected. Using local temp directory: {local_output_dir}")
  else:
    local_output_dir = args.output_dir

  fold_dir = os.path.join(local_output_dir, args.fold)
  os.makedirs(fold_dir, exist_ok=True)

  repo_id = "lmms-lab/LLaVA-Video-178K"

  # 1. Download and convert metadata
  print("Downloading metadata...")
  metadata_file = f"{args.fold}_cap_processed.json"
  local_metadata_path = hf_hub_download(
      repo_id=repo_id,
      filename=f"{args.fold}/{metadata_file}",
      repo_type="dataset",
      local_dir=local_output_dir,
      local_dir_use_symlinks=False,
  )

  print("Converting metadata to Parquet...")
  dataset = datasets.load_dataset("json", data_files=local_metadata_path, split="train")

  parquet_path = os.path.join(fold_dir, "llava-video-178k-caption-00000-of-00001.parquet")
  dataset.to_parquet(parquet_path)
  print(f"Saved parquet to {parquet_path}")

  # 2. Download and extract videos
  if args.download_videos:
    print("Downloading videos...")
    if args.video_subset is not None:
      tar_files = [f"{args.fold}/{args.fold}_videos_{i}.tar.gz" for i in range(1, args.video_subset + 1)]
    else:
      tar_files = [f"{args.fold}/{args.fold}_videos_{i}.tar.gz" for i in range(1, 9)]

    for tar_file in tar_files:
      print(f"Downloading {tar_file}...")
      local_tar_path = hf_hub_download(
          repo_id=repo_id,
          filename=tar_file,
          repo_type="dataset",
          local_dir=local_output_dir,
          local_dir_use_symlinks=False,
      )

      print(f"Extracting {local_tar_path} to {fold_dir}...")
      with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=fold_dir)

      # Optionally remove tar file to save space
      # os.remove(local_tar_path)

    print("Videos preparation complete.")
  else:
    print("Skipping videos download. Use --download_videos to download them.")

  if is_gcs:
    print(f"Uploading prepared dataset from {fold_dir} to {args.output_dir}...")

    # Ensure trailing slash for GCS destination to copy directory correctly
    gcs_dest = args.output_dir
    if not gcs_dest.endswith("/"):
      gcs_dest += "/"

    cmd = ["gsutil", "-m", "cp", "-r", fold_dir, gcs_dest]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
      print(f"Error uploading to GCS: {result.stderr}")
      raise RuntimeError(f"Failed to upload to GCS: {result.stderr}")
    print("Upload complete.")

    print(f"Cleaning up local temp directory {fold_dir}...")
    shutil.rmtree(fold_dir)
    print("Cleanup complete.")


if __name__ == "__main__":
  main()
