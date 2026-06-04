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


r"""Layerwise quantization for large models

Provides a utility to load and quantize a checkpoint layer by layer. Currently, it supports DeepSeek-family models only.

Example cmd:

python3 -m maxtext.utils.layerwise_quantization  src/maxtext/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH?} load_parameters_path=${LOAD_PARAMS_PATH?} \
  model_name=deepseek2-16b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 \
  ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=1 \
  attention=dot_product quantization=int8 async_checkpointing=false enable_single_controller=true \
  tokenizer_type=huggingface megablox=false sparse_matmul=false \
  save_quantized_params_path=${SAVE_PARAMS_PATH?} checkpoint_storage_use_ocdbt=False \
  checkpoint_storage_use_zarr3=False

"""

import os
from typing import Any, Sequence

from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from maxtext.common import common_types
from maxtext.layers import quantizations
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
import orbax.checkpoint as ocp
from maxtext.configs import pyconfig

PRNGKeyType = Any


class LayerwiseQuantization:
  """
  Layerwise quantization for large models.
  """

  def __init__(self, config: Any, rng: PRNGKeyType):
    self.config = config
    self.rng = rng

    # Mesh definition
    devices_array = maxtext_utils.create_device_mesh(config=config)
    self._mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

    self.quant = quantizations.configure_quantization(config)
    self.unboxed_abstract_state = None

  def load_and_quantize(self) -> None:
    """
    Load parameters layer by layer and quantize them.
    """
    self._load_and_quantize_nnx()

  def _load_and_quantize_nnx(self) -> None:
    """Whole-model NNX convert: load full-precision via TRAIN-mode `from_pretrained`,
    transfer kernels into a fresh CONVERT-mode model, run a forward (the
    `ToNNX(AqtDotGeneral)` bridge auto-captures `qrhs.frozen`), strip kernels at
    quantized paths, and save the serve-mode-shaped state.

    Two-step load: input checkpoints are typically full-precision (no AQT state
    on disk), so we can't `from_pretrained(quant_mode_str="convert")` directly —
    orbax would fail to find the missing `qrhs.frozen` leaves. Instead we load
    in TRAIN mode (which has only kernels), then copy them into a randomly
    initialized CONVERT model that already has the AQT variables provisioned.
    """
    config = self.config
    # MODEL_MODE_TRAIN avoids the PREFILL/AUTOREGRESSIVE cache plumbing — AQT
    # layers populate `qrhs.frozen` regardless of model_mode, so train mode is
    # simpler and faster.
    max_logging.log("Loading full-precision NNX checkpoint in TRAIN mode...")
    with self._mesh:
      train_model = model_creation_utils.from_pretrained(
          config,
          mesh=self._mesh,
          model_mode=common_types.MODEL_MODE_TRAIN,
          quant_mode_str="train",
      )

    max_logging.log("Building CONVERT-mode model (random init) and copying kernels in...")
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=self.rng)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      convert_model = model_creation_utils.from_config(
          config,
          mesh=self._mesh,
          rngs=rngs,
          model_mode=common_types.MODEL_MODE_TRAIN,
          quant_mode_str="convert",
      )
    self._copy_kernel_leaves_(convert_model, train_model)
    del train_model

    # Forward populates AqtDotGeneral_0.qrhs.frozen on every quantized layer.
    L = config.max_target_length
    decoder_input_tokens = jnp.zeros((1, L), dtype=jnp.int32)
    decoder_positions = jnp.arange(L, dtype=jnp.int32)[None, :]
    decoder_segment_ids = jnp.ones((1, L), dtype=jnp.int32)
    max_logging.log("Running CONVERT-mode forward to populate AQT scale factors...")
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      _ = convert_model(
          decoder_input_tokens,
          decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_TRAIN,
      )

    # Convert-mode state has both `kernel` (full precision) and `AqtDotGeneral_0.qrhs.frozen`
    # at every quantized DenseGeneral; the serve-mode reader expects only the latter.
    convert_state = nnx.state(convert_model).to_pure_dict()
    serve_state = self._strip_kernels_at_quantized_paths(convert_state)

    if config.save_quantized_params_path:
      max_logging.log(f"Saving NNX-format quantized checkpoint to {config.save_quantized_params_path}")

      # Wrap each leaf in `{"value": <array>}` so the on-disk shape matches what
      # `from_pretrained`'s NNX-detection branch reads back (it later does
      # `tree.map(lambda v: v["value"], ...)` on each leaf). Save directly via
      # orbax — `save_params_to_path` would add an outer `{"params": ...}` wrap
      # that the NNX path doesn't expect.
      def _wrap_value(node):
        if isinstance(node, dict):
          return {k: _wrap_value(v) for k, v in node.items()}
        return {"value": node}

      wrapped = _wrap_value(serve_state)
      orbax_checkpointer = ocp.PyTreeCheckpointer(
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
      )
      orbax_checkpointer.save(config.save_quantized_params_path, wrapped, force=True)
      max_logging.log(f"Saved NNX-format quantized checkpoint at: {config.save_quantized_params_path}")
    else:
      max_logging.log("Skipping save: save_quantized_params_path is null.")

  @staticmethod
  def _copy_kernel_leaves_(dst_model, src_model):
    """Copy the full-precision parameter leaves (kernel/embedding/scale/bias)
    from src into dst, leaving dst's AQT and RNG variables untouched.
    """
    src_dict = nnx.state(src_model).to_pure_dict()
    dst_state = nnx.state(dst_model)
    dst_dict = dst_state.to_pure_dict()

    def walk(d_node, s_node):
      if not (isinstance(d_node, dict) and isinstance(s_node, dict)):
        return
      for key, d_child in d_node.items():
        if key not in s_node:
          continue
        s_child = s_node[key]
        if key in ("kernel", "embedding", "scale", "bias") and not isinstance(d_child, dict):
          d_node[key] = s_child
        elif isinstance(d_child, dict):
          walk(d_child, s_child)

    walk(dst_dict, src_dict)
    nnx.replace_by_pure_dict(dst_state, dst_dict)
    nnx.update(dst_model, dst_state)

  @staticmethod
  def _strip_kernels_at_quantized_paths(state_dict):
    """Drop `kernel` keys at any node that has a sibling `AqtDotGeneral_0`.

    In convert mode each quantized DenseGeneral keeps both the full-precision
    `kernel` (an nnx.Param) and the AQT-quantized `AqtDotGeneral_0.qrhs.frozen`
    side-by-side. Serve mode (the on-disk shape `from_pretrained` reads back)
    only carries the latter; the kernel is recreated as a dummy zero in
    `linears.DenseGeneral.__call__`.
    """
    if not isinstance(state_dict, dict):
      return state_dict
    has_aqt = "AqtDotGeneral_0" in state_dict
    out = {}
    for k, v in state_dict.items():
      if k == "kernel" and has_aqt:
        continue
      out[k] = LayerwiseQuantization._strip_kernels_at_quantized_paths(v) if isinstance(v, dict) else v
    return out


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  validate_config(config)
  max_utils.print_system_information()
  rng = jax.random.PRNGKey(1234)
  quantization = LayerwiseQuantization(config, rng)
  quantization.load_and_quantize()


def validate_config(config):
  assert (
      config.load_full_state_path == ""
  ), "Operation on full states not supported! Convert to parameter checkpoint first."


if __name__ == "__main__":
  app.run(main)
