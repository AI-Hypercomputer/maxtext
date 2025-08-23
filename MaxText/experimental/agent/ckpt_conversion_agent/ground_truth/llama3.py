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
def LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False, saving_to_hf=False):
    """Creates parameter transformation functions for converting between MaxText and
    HuggingFace formats.
    This function generates a mapping of transformation functions that handle the necessary
    conversions between MaxText and HuggingFace parameter formats, including operations like
    reshaping.
    """
    nlayers = config["num_hidden_layers"]

    def scale_query_layer(input_tensor, target_shape):
      def to_hf():
          depth_scale = np.dtype("float32").type(np.sqrt(config["head_dim"]))

          original_dtype = input_tensor.dtype
          output_tensor = input_tensor.astype(np.float32) * depth_scale
          return output_tensor.astype(original_dtype)

      def from_hf():
          depth_scale = np.dtype("float32").type(1 / np.sqrt(config["head_dim"]))

          original_dtype = input_tensor.dtype
          output_tensor = input_tensor.astype(np.float32) * depth_scale
          return output_tensor.astype(original_dtype)
      if saving_to_hf:
          return to_hf()
      else:
          return from_hf()

    def adjust_rope(input_tensor, target_shape):
      def from_hf(arr):
          """Convert from HF's concatenated layout to MaxText's interleaved layout"""
          half_dim = arr.shape[-1] // 2
          first_half = arr[..., :half_dim]
          second_half = arr[..., half_dim:]
          return jax.numpy.stack([first_half, second_half], axis=-1).reshape(arr.shape)

      def to_hf(arr):
          """Convert from MaxText's interleaved layout to HF's concatenated layout"""
          evens = arr[..., ::2]
          odds = arr[..., 1::2]
          return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)

      if saving_to_hf:
          return to_hf(input_tensor)
      else:
          return from_hf(input_tensor)


    def reshape_kernel(input_tensor, target_shape):
      def to_hf():
          flipped_target_shape = np.flip(np.array(target_shape))
          return input_tensor.reshape(flipped_target_shape).transpose()

      def from_hf():
          return input_tensor.transpose().reshape(target_shape)

      if saving_to_hf:
          return to_hf()
      else:
          return from_hf()

    query_hooks = [reshape_kernel, adjust_rope, scale_query_layer]
    key_hooks = [reshape_kernel, adjust_rope]

    if not saving_to_hf:
        query_hooks.reverse()
        key_hooks.reverse()

    hook_fns = {}

    hook_fns["params-decoder-logits_dense-kernel"] = reshape_kernel

    if scan_layers:
        hook_fns = {
            **hook_fns,
            f"params-decoder-layers-self_attention-query-kernel": query_hooks,
            f"params-decoder-layers-self_attention-key-kernel": key_hooks,
            f"params-decoder-layers-self_attention-value-kernel": reshape_kernel,
            f"params-decoder-layers-self_attention-out-kernel": reshape_kernel,
            f"params-decoder-layers-mlp-wi_0-kernel": reshape_kernel,
            f"params-decoder-layers-mlp-wi_1-kernel": reshape_kernel,
            f"params-decoder-layers-mlp-wo-kernel": reshape_kernel,
        }
    else:
        for layer_idx in range(nlayers):
            hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-query-kernel"] = query_hooks
            hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-key-kernel"] = key_hooks
            hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-value-kernel"] = reshape_kernel
            hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-out-kernel"] = reshape_kernel
            hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wi_0-kernel"] = reshape_kernel 
            hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wi_1-kernel"] = reshape_kernel 
            hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wo-kernel"] = reshape_kernel 
    return hook_fns