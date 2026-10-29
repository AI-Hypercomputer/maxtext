# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan 2.1 model family package."""

# TODO: Known deviations from the m3 rules (m3 infra is still in progress); refactor planned:
#   * The sharding strategies are yml rule tables (`*_logical_axis_rules` in the config) in the legacy logical-axis
#     vocabulary, on Wan's own mesh axes (`context`; the VAE's `redundant` / `vae_spatial` reshape of the devices)
#     rather than m3's `MESH_AXIS_NAMES`.
#   * The VAE causal-conv kernel is replicated (its legacy `vae_spatial` spec never resolved).
#   * The transformer's activation constraints (model / attention inputs, attention output) are off: they were no-ops
#     under the legacy `with mesh:` and re-enabling them changes numerics.
#   * `transformer_wan.py` queries the TPU generation (`utils_wan.get_tpu_type`) to pick a 128 / 256 padding alignment.
#   * `utils_wan.py` also holds runtime helpers (mesh creation, prompt loading, config parsing).
#   * Model code logs through `maxtext.utils.max_logging`.
#   * Flash attention calls Tokamax's Splash kernel modules (`tokamax._src`) with Wan's own padding / segment-id /
#     shard_map glue: the public `tokamax.dot_product_attention` Mosaic TPU path needs the sequence to divide its
#     blocks (Wan 720p has 75600 tokens) and has no padding mask.
# Attention kernels: `tokamax.dot_product_attention` (XLA) and Tokamax's Splash TPU kernel, in `transformer_wan.py`.
from .autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
from .text_encoder_wan import UMT5EncoderModel
from .transformer_wan import WanModel
from .utils_wan import VAE_MESH_AXES, WAN_MESH_AXES, WanSharding
