# Copyright 2025 Google LLC
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

"""General Hugging Face Diffusers checkpoint converter and manager for MaxText.

This module acts as the universal entrypoint for converting any Diffusers model to MaxText:
1. Downloads and verifies general Diffusers pipeline components (VAE, text encoders,
   tokenizers, schedulers, model_index.json) to ensure the complete pipeline is available.
2. Identifies the diffusion model architecture/family.
3. Dispatches transformer/UNet weights conversion to the model-specific converter
   (`convert_<model_type>.py`, e.g. `convert_wan.py`).
"""

import importlib
import json
import os
from typing import Any, Optional, Sequence

from absl import app
from huggingface_hub import snapshot_download

from maxtext.utils import max_logging


def download_diffusers_components(
    repo_id: str,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
) -> str:
  """Downloads general diffusers pipeline components (VAE, text encoder, tokenizer, scheduler).

  By default, downloads all non-transformer components (VAE, text encoders, tokenizers,
  schedulers, and metadata configs) so they are fully cached and ready for inference,
  leaving large transformer/UNet weights conversion to model-specific converters.

  Args:
    repo_id: Hugging Face model repository ID or local directory path.
    token: Optional Hugging Face auth token.
    cache_dir: Optional cache directory.
    revision: Optional repository git revision.
    components: Optional list of component folder prefixes to include.
                Defaults to ("vae", "text_encoder", "tokenizer", "scheduler").

  Returns:
    The local path to the downloaded repository snapshot or directory.
  """
  if os.path.isdir(repo_id):
    max_logging.log(f"Model path '{repo_id}' is a local directory. Skipping HF download.")
    return repo_id

  if components is None:
    allow_patterns = [
        "vae/*",
        "text_encoder*/*",
        "tokenizer*/*",
        "scheduler/*",
        "model_index.json",
        "*.json",
    ]
  else:
    allow_patterns = [f"{c}*/*" for c in components] + ["model_index.json", "*.json"]

  max_logging.log(
      f"Downloading/verifying general Diffusers components for '{repo_id}' "
      f"(patterns: {allow_patterns})..."
  )
  try:
    local_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        cache_dir=cache_dir,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    max_logging.log(f"General Diffusers components verified in cache at: {local_dir}")
    return local_dir
  except Exception as e:
    max_logging.log(f"Online download failed ({e}), attempting local cache fallback...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        cache_dir=cache_dir,
        revision=revision,
        allow_patterns=allow_patterns,
        local_files_only=True,
    )
    max_logging.log(f"Found cached Diffusers components at: {local_dir}")
    return local_dir


def get_model_type(config: Optional[Any], model_name_or_path: str) -> str:
  """Resolves the diffusion model architecture/family for dispatching to convert_<model_type>.py."""
  # 1. Inspect config.model_name first (e.g. wan2.1 -> wan, flux-dev -> flux, etc.)
  if config is not None and getattr(config, "model_name", None):
    m_name = str(config.model_name).lower()
    if "wan" in m_name:
      return "wan"
    if "flux" in m_name:
      return "flux"
    if "sd3" in m_name:
      return "sd3"
    if "cogvideo" in m_name:
      return "cogvideox"

  # 2. Check repository or path name (e.g. Wan-AI/Wan2.1-T2V-14B-Diffusers)
  norm_name = model_name_or_path.lower()
  if "wan" in norm_name:
    return "wan"
  if "flux" in norm_name:
    return "flux"
  if "sd3" in norm_name or "stable-diffusion-3" in norm_name:
    return "sd3"
  if "cogvideo" in norm_name:
    return "cogvideox"

  # 3. Check explicit model_family or model_type if provided
  if config is not None:
    if getattr(config, "model_family", None):
      return str(config.model_family).lower().strip()
    if getattr(config, "model_type", None) and str(config.model_type).lower() not in ("t2v", "i2v", "t2i", "i2i"):
      return str(config.model_type).lower().strip()

  # Fallback to repository name prefix
  return norm_name.split("/")[-1].split("-")[0]


def get_converter_module(model_type: str):
  """Dynamically imports the converter module for the specified diffusion model type."""
  module_name = f"maxtext.checkpoint_conversion.convert_{model_type}"
  try:
    return importlib.import_module(module_name)
  except ModuleNotFoundError as e:
    raise NotImplementedError(
        f"Could not find checkpoint converter module '{module_name}'. "
        f"To support model type '{model_type}', create 'src/maxtext/checkpoint_conversion/convert_{model_type}.py' "
        f"implementing a `convert_checkpoint` function."
    ) from e


def convert_diffusers_checkpoint(
    model_name: str,
    config: Optional[Any] = None,
    output_dir: Optional[str] = None,
    download_components: bool = True,
    **kwargs,
) -> str:
  """Universal Diffusers checkpoint conversion function.

  1. Downloads general Diffusers components (VAE, text encoder, tokenizer, scheduler).
  2. Dispatches model-specific transformer/UNet conversion to convert_<model_type>.py.

  Args:
    model_name: Hugging Face model repository ID or local path.
    config: Optional MaxText configuration.
    output_dir: Target directory for the converted Orbax checkpoint.
    download_components: Whether to download and verify general diffusers components.
    **kwargs: Additional parameters passed to the model-specific converter.

  Returns:
    Path to the converted Orbax checkpoint.
  """
  if download_components:
    token = (
        getattr(config, "hf_access_token", None)
        or getattr(config, "hf_token", None)
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_AUTH_TOKEN")
    )
    cache_dir = os.environ.get(
        "CACHE_DIR",
        os.path.join(os.environ.get("HF_HOME", "/dev/shm/hf_cache"), "maxdiffusion"),
    )
    try:
      download_diffusers_components(
          repo_id=model_name,
          token=token,
          cache_dir=cache_dir,
      )
    except Exception as e:
      max_logging.log(f"Notice: General diffusers component download encountered: {e}. Proceeding to model conversion.")

  model_type = get_model_type(config, model_name)
  max_logging.log(f"Dispatching checkpoint conversion for model '{model_name}' to convert_{model_type}.py...")
  converter_module = get_converter_module(model_type)

  converter_fn = getattr(
      converter_module,
      "convert_checkpoint",
      getattr(converter_module, "checkpoint_conversion", None),
  )
  if converter_fn is None:
    raise AttributeError(
        f"Converter module {converter_module.__name__} does not define convert_checkpoint or checkpoint_conversion."
    )

  return converter_fn(model_name=model_name, config=config, output_dir=output_dir, **kwargs)


# Aliases for backward compatibility
convert_checkpoint = convert_diffusers_checkpoint
checkpoint_conversion = convert_diffusers_checkpoint


def main(argv: Sequence[str]) -> None:
  from maxtext.configs import pyconfig
  pyconfig.initialize(argv)
  config = pyconfig.config

  output_dir = (
      getattr(config, "base_output_directory", None)
      or getattr(config, "checkpoint_directory", None)
      or getattr(config, "output_dir", None)
      or getattr(config, "checkpoint_dir", None)
  )
  if not output_dir:
    raise ValueError("base_output_directory or checkpoint_directory must be provided.")

  model_name = getattr(config, "pretrained_model_name_or_path", None) or getattr(config, "model_name", None)
  if not model_name:
    raise ValueError("pretrained_model_name_or_path must be specified in config or via CLI.")

  checkpoint_path = convert_diffusers_checkpoint(
      model_name=model_name,
      config=config,
      output_dir=output_dir,
  )
  max_logging.log(f"Diffusers conversion complete! Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
  app.run(main)
