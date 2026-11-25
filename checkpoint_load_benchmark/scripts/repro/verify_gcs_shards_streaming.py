# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streams real production DeepSeek-V3 and DeepSeek-V4 GCS shards vs offline BF16 shards.

Checks:
  1. `deepseek3-671b` shard 1 (`FP8` + `F32` `*_scale_inv`, including non-divisible `576x7168`
     `kv_a_proj_with_mqa.weight`) streamed and dequantized on the fly vs offline `BF16` shard 1.
  2. `deepseek4-284b` shard 5 (`FP8` + `F8_E8M0` `*.scale` and packed `MXFP4` `I8` + `F8_E8M0` `*.scale`)
     streamed and dequantized on the fly vs offline `BF16` shard 5.
"""

import argparse
import os
import sys
import types

if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=4".strip()

# pylint: disable=wrong-import-position
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.checkpoint_conversion.utils import safetensors_reader
import numpy as np

DEFAULT_GCS_ROOT = "gs://mesa-maxtext-europe-west4/mesa-bench-v2"


def _reshape_transpose(weight, shape):
  return weight.T.reshape(shape)


def _ones_hook(weight, shape):
  del weight
  return np.ones(shape, dtype=np.float32)


def _verify_deepseek3_shard(gcs_root: str, mesh: Mesh, cfg: types.SimpleNamespace) -> None:
  """Verifies DeepSeek-V3 shard 1 FP8 + F32 `_scale_inv` streaming dequantization against offline BF16."""
  ds3_raw = f"{gcs_root}/hf_cache/deepseek3-671b/model-00001-of-000163.safetensors"
  ds3_bf16 = f"{gcs_root}/hf_cache_bf16/deepseek3-671b/model-00001-of-000163.safetensors"
  ds3_target = {
      "decoder": {
          "dense_layers_0": {
              "pre_self_attention_layer_norm": {
                  "scale": jax.ShapeDtypeStruct((7168,), jnp.bfloat16, sharding=NamedSharding(mesh, P("fsdp")))
              },
              "self_attention": {
                  "wkv_a": {
                      "kernel": jax.ShapeDtypeStruct(
                          (7168, 576), jnp.bfloat16, sharding=NamedSharding(mesh, P("fsdp", "tensor"))
                      )
                  },
                  "wq_a": {
                      "kernel": jax.ShapeDtypeStruct(
                          (7168, 1536), jnp.bfloat16, sharding=NamedSharding(mesh, P("fsdp", "tensor"))
                      )
                  },
              },
          }
      }
  }
  ds3_pmap = {
      "params-decoder-dense_layers_0-pre_self_attention_layer_norm-scale": "model.layers.0.input_layernorm.weight",
      "params-decoder-dense_layers_0-self_attention-wkv_a-kernel": "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
      "params-decoder-dense_layers_0-self_attention-wq_a-kernel": "model.layers.0.self_attn.q_a_proj.weight",
  }
  ds3_hmap = {
      "params-decoder-dense_layers_0-self_attention-wkv_a-kernel": _reshape_transpose,
      "params-decoder-dense_layers_0-self_attention-wq_a-kernel": _reshape_transpose,
  }
  got_ds3 = hf_streaming_load.load_hf_params_streaming(ds3_raw, ds3_target, ds3_pmap, ds3_hmap, cfg)["params"]
  want_ds3 = hf_streaming_load.load_hf_params_streaming(ds3_bf16, ds3_target, ds3_pmap, ds3_hmap, cfg)["params"]
  for name, got_leaf, want_leaf in [
      (
          "norm",
          got_ds3["decoder"]["dense_layers_0"]["pre_self_attention_layer_norm"]["scale"],
          want_ds3["decoder"]["dense_layers_0"]["pre_self_attention_layer_norm"]["scale"],
      ),
      (
          "wkv_a (7168x576 non-divisible FP8)",
          got_ds3["decoder"]["dense_layers_0"]["self_attention"]["wkv_a"]["kernel"],
          want_ds3["decoder"]["dense_layers_0"]["self_attention"]["wkv_a"]["kernel"],
      ),
      (
          "wq_a (7168x1536 FP8)",
          got_ds3["decoder"]["dense_layers_0"]["self_attention"]["wq_a"]["kernel"],
          want_ds3["decoder"]["dense_layers_0"]["self_attention"]["wq_a"]["kernel"],
      ),
  ]:
    eq = bool(jnp.array_equal(got_leaf, want_leaf))
    print(f"  [Real DS3-671B GCS] {name}: shape={got_leaf.shape}, dtype={got_leaf.dtype}, bit_exact={eq}")
    assert eq, f"Mismatch on {name}"


def _verify_deepseek4_shard(gcs_root: str, mesh: Mesh, cfg: types.SimpleNamespace) -> None:
  """Verifies DeepSeek-V4 shard 5 FP8 + MXFP4 + F8_E8M0 streaming dequantization against offline BF16."""
  ds4_raw = f"{gcs_root}/hf_cache/deepseek4-284b/model-00005-of-00046.safetensors"
  ds4_bf16 = f"{gcs_root}/hf_cache_bf16/deepseek4-284b/model-00005-of-00046.safetensors"
  with safetensors_reader.SafetensorsReader(ds4_raw) as reader:
    md4 = reader.metadata
  fp8_pair = next(
      k
      for k, v in md4.items()
      if k.endswith(".weight") and v.dtype == np.dtype(jnp.float8_e4m3fn) and f"{k[:-7]}.scale" in md4
  )
  fp4_experts = [
      f"layers.3.ffn.experts.{e}.w1.weight"
      for e in range(2)
      if f"layers.3.ffn.experts.{e}.w1.weight" in md4 and f"layers.3.ffn.experts.{e}.w1.scale" in md4
  ]
  fp8_shape_mt = (md4[fp8_pair].shape[1], md4[fp8_pair].shape[0])
  fp4_shape_mt = (len(fp4_experts), md4[fp4_experts[0]].shape[1] * 2, md4[fp4_experts[0]].shape[0])

  ds4_target = {
      "decoder": {
          "layers_3": {
              "mhc_attention": {
                  "mhc_norm": {
                      "scale": jax.ShapeDtypeStruct((16384,), jnp.bfloat16, sharding=NamedSharding(mesh, P("fsdp")))
                  }
              },
              "self_attention": {
                  "proj": {
                      "kernel": jax.ShapeDtypeStruct(
                          fp8_shape_mt, jnp.bfloat16, sharding=NamedSharding(mesh, P("fsdp", "tensor"))
                      )
                  }
              },
              "mlp": {
                  "MoeBlock_0": {
                      "wi_0": jax.ShapeDtypeStruct(
                          fp4_shape_mt, jnp.bfloat16, sharding=NamedSharding(mesh, P("fsdp", "tensor", None))
                      )
                  }
              },
          }
      }
  }
  ds4_pmap = {
      "params-decoder-layers_3-mhc_attention-mhc_norm-scale": None,
      "params-decoder-layers_3-self_attention-proj-kernel": fp8_pair,
      "params-decoder-layers_3-mlp-MoeBlock_0-wi_0": fp4_experts,
  }
  ds4_hmap = {
      "params-decoder-layers_3-mhc_attention-mhc_norm-scale": _ones_hook,
      "params-decoder-layers_3-self_attention-proj-kernel": _reshape_transpose,
      "params-decoder-layers_3-mlp-MoeBlock_0-wi_0": _reshape_transpose,
  }
  got_ds4 = hf_streaming_load.load_hf_params_streaming(ds4_raw, ds4_target, ds4_pmap, ds4_hmap, cfg)["params"]
  want_ds4 = hf_streaming_load.load_hf_params_streaming(ds4_bf16, ds4_target, ds4_pmap, ds4_hmap, cfg)["params"]
  for name, got_leaf, want_leaf in [
      (
          "mhc_norm (None source)",
          got_ds4["decoder"]["layers_3"]["mhc_attention"]["mhc_norm"]["scale"],
          want_ds4["decoder"]["layers_3"]["mhc_attention"]["mhc_norm"]["scale"],
      ),
      (
          f"{fp8_pair} (FP8 + F8_E8M0)",
          got_ds4["decoder"]["layers_3"]["self_attention"]["proj"]["kernel"],
          want_ds4["decoder"]["layers_3"]["self_attention"]["proj"]["kernel"],
      ),
      (
          f"{fp4_experts} (Stacked MXFP4 I8 + F8_E8M0)",
          got_ds4["decoder"]["layers_3"]["mlp"]["MoeBlock_0"]["wi_0"],
          want_ds4["decoder"]["layers_3"]["mlp"]["MoeBlock_0"]["wi_0"],
      ),
  ]:
    eq = bool(jnp.array_equal(got_leaf, want_leaf))
    print(f"  [Real DS4-284B GCS] {name}: shape={got_leaf.shape}, dtype={got_leaf.dtype}, bit_exact={eq}")
    assert eq, f"Mismatch on {name}"


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint for real GCS production shard streaming parity checks."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--gcs-root",
      default=DEFAULT_GCS_ROOT,
      help="Regional GCS root containing hf_cache/ and hf_cache_bf16/.",
  )
  parser.add_argument(
      "--models",
      nargs="+",
      choices=["deepseek3-671b", "deepseek4-284b"],
      default=["deepseek3-671b", "deepseek4-284b"],
      help="Models whose GCS shards should be verified.",
  )
  args = parser.parse_args(argv)
  gcs_root = args.gcs_root.rstrip("/")
  mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "tensor"))
  cfg = types.SimpleNamespace(scan_layers=False, param_scan_axis=1)

  if "deepseek3-671b" in args.models:
    print(f"=== Streaming real DeepSeek-V3 shard 1 from {gcs_root} ===")
    _verify_deepseek3_shard(gcs_root, mesh, cfg)
  if "deepseek4-284b" in args.models:
    print(f"=== Streaming real DeepSeek-V4 shard 5 from {gcs_root} ===")
    _verify_deepseek4_shard(gcs_root, mesh, cfg)
  print("[PASS] All real GCS production shard streaming checks matched offline BF16 bit-for-bit.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
