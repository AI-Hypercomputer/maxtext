# -*- coding: utf-8 -*-
"""Step 1 of 5 maxtext multimodal alignment proof of concept project.

This file reads the omni-gemma3-qwen3.yml configuration and 
  1. download and convert a Vision-Language model (e.g. gemma3-4b) from hugging face to maxtext format
  2. download and convert an text-only model (e.g. qwen3-4b) from hugging face to maxtext format
  3. stitch the vision component and llm checkpoints into a single omni checkpoint
  4. save the stitched checkpoints to a output directory

Example usage:
python maxtext/experimental/omni_poc/prepare_checkpoint.py
"""
import os
import subprocess
import sys
from etils import epath
from maxtext.utils.globals import MAXTEXT_PKG_DIR


def main():

  # Hugging Face Token & Login. Can be set via HF_TOKEN environment variable.
  hf_token = os.environ.get("HF_TOKEN", "")

  # Base GCS or local directory where converted and stitched checkpoints will be stored.
  base_output_directory = "gs://YOUR_BUCKET_NAME/omni_checkpoints"

  if hf_token:
    try:
      from huggingface_hub import login  # pylint: disable=import-outside-toplevel

      login(token=hf_token)
    except ImportError:
      print("huggingface_hub not installed. Skipping login.")
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Hugging Face login failed: {e}")
      print("Ensure HF_TOKEN is valid if needed.\n")

  # Configuration & Paths
  vision_maxtext_model = "gemma3-4b"
  vision_hf_repo = "google/gemma-3-4b-it"

  llm_maxtext_model = "qwen3-4b"
  llm_hf_repo = "Qwen/Qwen3-4B"

  # Path to combined YAML configuration inside experimental/omni_poc
  omni_config_path = os.path.join(MAXTEXT_PKG_DIR, "experimental", "omni_poc", "omni-gemma3-qwen3.yml")

  vision_ckpt_dir = f"{base_output_directory}/{vision_maxtext_model}_converted"
  llm_ckpt_dir = f"{base_output_directory}/{llm_maxtext_model}_converted"
  stitched_ckpt_dir = f"{base_output_directory}/omni_stitched_{vision_maxtext_model}_{llm_maxtext_model}"

  vision_items_path = epath.Path(vision_ckpt_dir) / "0/items"
  llm_items_path = epath.Path(llm_ckpt_dir) / "0/items"
  stitched_items_path = epath.Path(stitched_ckpt_dir) / "0/items"

  print(f"Base Output Directory:  {base_output_directory}")
  print(f"Vision Converted Path:  {vision_items_path}")
  print(f"LLM Converted Path:     {llm_items_path}")
  print(f"Stitched Target Path:   {stitched_items_path}\n")

  env = os.environ.copy()
  env["JAX_PLATFORMS"] = "cpu"  # Conversion and stitching run smoothly on CPU/TPU

  # Step 1: Download & Convert Vision Model from Hugging Face -> MaxText
  print("=" * 60)
  if not vision_items_path.exists():
    print(f"Converting Vision Model ({vision_maxtext_model}) from Hugging Face ({vision_hf_repo})...")
    try:
      subprocess.run(
          [
              sys.executable,
              "-m",
              "maxtext.checkpoint_conversion.to_maxtext",
              os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
              f"model_name={vision_maxtext_model}",
              f"base_output_directory={vision_ckpt_dir}",
              "use_multimodal=True",
              "scan_layers=True",
              "skip_jax_distributed_system=True",
              "--eager_load_method=transformers",
              "--lazy_load_tensors=False",
              "log_config=False",
          ],
          check=True,
          env=env,
      )
      print("Vision checkpoint conversion successful!\n")
    except subprocess.CalledProcessError as e:
      print(f"Failed to convert Vision checkpoint: {e}")
      sys.exit(1)
  else:
    print(f"Step 1: Vision checkpoint already exists at {vision_items_path}")

  # Step 2: Download & Convert Language Model from Hugging Face -> MaxText
  print("=" * 60)
  if not llm_items_path.exists():
    print(f"Converting Language Model ({llm_maxtext_model}) from Hugging Face ({llm_hf_repo})...")
    try:
      subprocess.run(
          [
              sys.executable,
              "-m",
              "maxtext.checkpoint_conversion.to_maxtext",
              os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
              f"model_name={llm_maxtext_model}",
              f"base_output_directory={llm_ckpt_dir}",
              "scan_layers=True",
              "skip_jax_distributed_system=True",
              "--eager_load_method=transformers",
              "--lazy_load_tensors=False",
              "log_config=False",
          ],
          check=True,
          env=env,
      )
      print("LLM checkpoint conversion successful!\n")
    except subprocess.CalledProcessError as e:
      print(f"Failed to convert LLM checkpoint: {e}")
      sys.exit(1)
  else:
    print(f"Step 2: LLM checkpoint already exists at {llm_items_path}")

  # Step 3: Checkpoint Stitching (Vision Tower + LLM Decoder + Fresh Projector)
  print("=" * 60)
  print("Stitching Vision and LLM subtrees into unified Omni checkpoint...")
  try:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "maxtext.experimental.omni_poc.checkpoint_stitcher.stitch",
            omni_config_path,
            f"vision_load_path={str(vision_items_path)}",
            f"llm_load_path={str(llm_items_path)}",
            f"stitched_output_path={str(stitched_items_path)}",
        ],
        check=True,
        env=env,
    )
  except subprocess.CalledProcessError as e:
    print(f"Failed during checkpoint stitching: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
