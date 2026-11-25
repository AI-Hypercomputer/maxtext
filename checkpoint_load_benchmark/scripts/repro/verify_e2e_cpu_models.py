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

"""Local CPU end-to-end verification across reduced-dim DeepSeek-V4, Gemma-4, and GLM-5.2 models.

Verifies:
  1. Reduced-dim `deepseek4-284b` (`FP8` + `MXFP4` + `F8_E8M0`):
     - `from_pretrained` weights and deterministic forward-pass logits match offline `BF16` bit-for-bit
       across `scan_layers={True, False}` x `safetensors_dynamic_streaming={False, True}`.
     - `transform_hf_state_to_mt_state` empties `hf_state`, deletes 100% of raw quantized `jax.Array`
       weight & scale buffers on device, and leaves `0` restored model weight buffers deleted.
  2. Reduced-dim `gemma4-26b` (`scan_layers={True, False}`):
     - Shared `experts.gate_up_proj` composite splitting leaves `0` deleted restored buffers and matches
       `hf_streaming_load` bit-for-bit.
  3. Reduced-dim `glm5.2-744b` (`IndexShare` across `scan_layers=True` and `scan_layers=False` with
     `prune_shared_indexers={True, False}`):
     - Shared indexer weights leave `0` deleted restored buffers and match `hf_streaming_load` bit-for-bit.
"""

import argparse
import functools
import json
import os
import pathlib
import sys
import tempfile
from unittest import mock

if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=4".strip()

# pylint: disable=wrong-import-position
from flax import nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils
import numpy as np
import safetensors.flax

_DEFAULT_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[3])

_FP4_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

_DT_NAMES = {
    np.dtype(np.int8): "I8",
    np.dtype(np.int64): "I64",
    np.dtype(jnp.bfloat16): "BF16",
    np.dtype(np.float32): "F32",
    np.dtype(jnp.float8_e4m3fn): "F8_E4M3",
    np.dtype(jnp.float8_e8m0fnu): "F8_E8M0",
}


def _save_sharded_safetensors(tensors: dict, directory: str, num_files: int = 2) -> None:
  """Writes `tensors` (supporting `F8_E8M0` and `F8_E4M3` headers) into sharded `.safetensors` files."""
  keys = sorted(tensors)
  for i in range(num_files):
    sub = {k: tensors[k] for k in keys[i::num_files]}
    header, chunks, offset = {}, [], 0
    for name in sorted(sub):
      arr = np.asarray(sub[name])
      raw = np.ascontiguousarray(arr).view(np.uint8).tobytes()
      header[name] = {
          "dtype": _DT_NAMES[np.dtype(arr.dtype)],
          "shape": list(arr.shape),
          "data_offsets": [offset, offset + len(raw)],
      }
      chunks.append(raw)
      offset += len(raw)
    hb = json.dumps(header).encode("utf-8")
    shard_path = os.path.join(directory, f"model-{i + 1:05d}-of-{num_files:05d}.safetensors")
    with open(shard_path, "wb") as f:
      f.write(len(hb).to_bytes(8, "little") + hb)
      for chunk in chunks:
        f.write(chunk)


def _ref_fp8(weight: np.ndarray, scale: np.ndarray, block: int = 128) -> np.ndarray:
  rows, cols = weight.shape
  x = np.asarray(jnp.asarray(weight).astype(jnp.float32))
  sf = np.asarray(jnp.asarray(scale).astype(jnp.float32))
  s_exp = np.repeat(np.repeat(sf, block, axis=0), block, axis=1)[:rows, :cols]
  return np.asarray(jnp.asarray(x * s_exp).astype(jnp.bfloat16))


def _ref_mxfp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
  u8 = np.ascontiguousarray(packed).view(np.uint8)
  unpacked = np.stack([_FP4_LUT[u8 & 0xF], _FP4_LUT[(u8 >> 4) & 0xF]], axis=-1)
  qf = unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])
  rows, cols = qf.shape[-2:]
  s_rows, s_cols = scale.shape[-2:]
  sf = np.asarray(jnp.asarray(scale).astype(jnp.float32))
  out = qf.reshape(-1, s_rows, rows // s_rows, s_cols, cols // s_cols) * sf.reshape(-1, s_rows, 1, s_cols, 1)
  return np.asarray(jnp.asarray(out).astype(jnp.bfloat16)).reshape(qf.shape)


def verify_deepseek4(repo_root: str) -> None:
  """Verifies reduced-dim `deepseek4-284b` quantized vs offline BF16 weights, buffer freeing, and forward logits."""
  print("=== [1/3] Verifying reduced-dim deepseek4-284b (FP8 + MXFP4 + F8_E8M0) ===")
  base_yml = os.path.join(repo_root, "src/maxtext/configs/base.yml")
  hf_cfg_dict = {
      "hidden_size": 32,
      "vocab_size": 64,
      "num_hidden_layers": 3,
      "q_lora_rank": 32,
      "o_lora_rank": 32,
      "o_groups": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 16,
      "moe_intermediate_size": 32,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "hc_mult": 2,
      "num_hash_layers": 1,
      "index_n_heads": 2,
      "index_head_dim": 16,
      "compress_ratios": [0, 128, 4],
      "mtp_num_layers": 1,
  }
  rng = np.random.default_rng(20261120)
  quant_hf, bf16_hf = {}, {}
  for name, shape_list in hf_shape.DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE(hf_cfg_dict).items():
    if not isinstance(name, str) or name == "layers.0.ffn.gate.bias":
      continue
    shape = tuple(shape_list)
    if name.endswith(".tid2eid"):
      quant_hf[name] = bf16_hf[name] = rng.integers(0, 4, size=shape, dtype=np.int64)
    elif any(s in name for s in ("_fn", "_base", "_scale", "attn_sink", ".ape", "gate.bias")):
      quant_hf[name] = bf16_hf[name] = rng.standard_normal(shape, dtype=np.float32) * 0.05
    elif any(s in name for s in (".ffn.experts.", ".ffn.shared_experts.", ".e_proj.weight")):
      rows, cols = shape[0], shape[1]
      packed = rng.integers(-128, 127, size=(rows, cols // 2), dtype=np.int8)
      s_u8 = jnp.asarray(rng.integers(122, 127, size=(rows, cols // 32), dtype=np.uint8))
      s_e8m0 = np.asarray(jax.lax.bitcast_convert_type(s_u8, jnp.float8_e8m0fnu))
      quant_hf[name] = packed
      quant_hf[f"{name[:-7]}.scale"] = s_e8m0
      bf16_hf[name] = _ref_mxfp4(packed, s_e8m0)
    elif len(shape) == 2 and not any(
        s in name for s in ("embed.weight", "head.weight", "ffn.gate.weight", "weights_proj.weight")
    ):
      w = np.asarray(jnp.asarray(rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(jnp.float8_e4m3fn))
      s_u8 = jnp.asarray(rng.integers(123, 127, size=(1, 1), dtype=np.uint8))
      s_e8m0 = np.asarray(jax.lax.bitcast_convert_type(s_u8, jnp.float8_e8m0fnu))
      quant_hf[name] = w
      quant_hf[f"{name[:-7]}.scale"] = s_e8m0
      bf16_hf[name] = _ref_fp8(w, s_e8m0, 128)
    else:
      quant_hf[name] = bf16_hf[name] = np.asarray(
          jnp.asarray(rng.standard_normal(shape, dtype=np.float32) * 0.05).astype(jnp.bfloat16)
      )

  fake_hf_cfg = mock.MagicMock()
  fake_hf_cfg.to_dict.return_value = hf_cfg_dict

  with tempfile.TemporaryDirectory() as qdir, tempfile.TemporaryDirectory() as bdir:
    _save_sharded_safetensors(quant_hf, qdir, num_files=3)
    _save_sharded_safetensors(bf16_hf, bdir, num_files=3)

    tokens = jnp.tile(jnp.array([[1, 5, 9, 12, 3, 7, 2, 4]], dtype=jnp.int32), (4, 1))
    pos = jnp.tile(jnp.arange(8, dtype=jnp.int32)[None, :], (4, 1))

    for scan_layers in (True, False):
      logits_by_mode = {}
      params_by_mode = {}
      last_cfg = None
      last_model = None
      for label, path, streaming in (
          ("quant_default", qdir, False),
          ("quant_streaming", qdir, True),
          ("bf16_ref", bdir, True),
      ):
        argv = [
            "",
            base_yml,
            "model_name=deepseek4-284b",
            "override_model_config=True",
            "base_emb_dim=32",
            "base_num_query_heads=2",
            "base_num_kv_heads=1",
            "head_dim=16",
            "qk_rope_head_dim=8",
            "q_lora_rank=32",
            "o_lora_rank=32",
            "o_groups=2",
            "base_mlp_dim=32",
            "base_moe_mlp_dim=32",
            "base_num_decoder_layers=3",
            "first_num_hash_layers=1",
            "num_experts=4",
            "num_experts_per_tok=2",
            "mhc_expansion_rate=2",
            "indexer_n_heads=2",
            "indexer_head_dim=16",
            "indexer_topk=4",
            "compress_ratios=[0,128,4]",
            "vocab_size=64",
            "mtp_num_layers=1",
            "attention=dot_product",
            f"scan_layers={str(scan_layers).lower()}",
            "source_checkpoint_layout=safetensors_dynamic",
            f"safetensors_dynamic_streaming={str(streaming).lower()}",
            f"load_parameters_path={path}",
            f"base_output_directory={bdir}",
            "weight_dtype=bfloat16",
            "dtype=bfloat16",
            "per_device_batch_size=1",
            "max_target_length=8",
            "ici_fsdp_parallelism=4",
            "ici_tensor_parallelism=1",
            "skip_jax_distributed_system=True",
        ]
        cfg = pyconfig.initialize(argv)
        object.__setattr__(cfg, "compress_ratios", [0, 128, 4])
        with mock.patch.dict(hf_model_configs.HF_MODEL_CONFIGS, {"deepseek4-284b": fake_hf_cfg}):
          model, _ = model_creation_utils.from_pretrained(cfg)
        last_cfg, last_model = cfg, model
        _, state = nnx.split(model)
        flat_live = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep=".")
        for leaf_name, leaf_val in flat_live.items():
          if isinstance(leaf_val, jax.Array) and not jax.dtypes.issubdtype(leaf_val.dtype, jax.dtypes.prng_key):
            assert not leaf_val.is_deleted(), f"{label} {leaf_name} buffer was deleted!"
        params_by_mode[label] = jax.device_get(flat_live)
        logits = model(
            tokens,
            pos,
            enable_dropout=False,
            decoder_target_tokens=tokens,
            decoder_target_mask=jnp.ones_like(tokens),
        )
        if isinstance(logits, tuple):
          logits = logits[0]
        logits_by_mode[label] = np.asarray(logits, dtype=np.float32)

      # Verify raw quantized weight & companion scale buffers are freed during transform_hf_state_to_mt_state.
      pure_weights = nnx.to_pure_dict(
          checkpointing._abstract_params(nnx.state(last_model))  # pylint: disable=protected-access
      )
      target_tree = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), pure_weights)
      pm = param_mapping.PARAM_MAPPING["deepseek4-284b"](hf_cfg_dict, last_cfg, scan_layers=scan_layers)
      hm = param_mapping.HOOK_FNS["deepseek4-284b"](hf_cfg_dict, last_cfg, scan_layers=scan_layers, saving_to_hf=False)
      hf_state = load_dynamic.rename_legacy_hf_keys(load_dynamic.load_sharded_hf_state(qdir), pm)
      quant_scale_buffers = [
          arr
          for key, arr in hf_state.items()
          if isinstance(arr, jax.Array) and (key.endswith(".scale") or arr.dtype in (jnp.float8_e4m3fn, jnp.int8))
      ]
      restored_direct = load_dynamic.transform_hf_state_to_mt_state(
          hf_state, target_tree, pm, hm, last_cfg, ckpt_dir=qdir
      )
      assert not hf_state, f"Expected empty hf_state after transform, left: {list(hf_state)[:5]}"
      assert all(arr.is_deleted() for arr in quant_scale_buffers), "Expected all quantized/scale buffers to be freed!"
      for k, arr in flax.traverse_util.flatten_dict(restored_direct["params"], sep=".").items():
        assert isinstance(arr, jax.Array) and not arr.is_deleted(), f"Restored leaf {k} deleted!"

      checked = 0
      for k, v_ref in params_by_mode["bf16_ref"].items():
        if jax.dtypes.issubdtype(v_ref.dtype, jax.dtypes.prng_key):
          continue
        checked += 1
        np.testing.assert_array_equal(
            params_by_mode["quant_default"][k], v_ref, err_msg=f"default mismatch at {k} (scan={scan_layers})"
        )
        np.testing.assert_array_equal(
            params_by_mode["quant_streaming"][k], v_ref, err_msg=f"streaming mismatch at {k} (scan={scan_layers})"
        )

      np.testing.assert_array_equal(logits_by_mode["quant_default"], logits_by_mode["bf16_ref"])
      np.testing.assert_array_equal(logits_by_mode["quant_streaming"], logits_by_mode["bf16_ref"])
      print(
          f"  [PASS] deepseek4-284b scan_layers={scan_layers}: {checked} weight leaves, {len(quant_scale_buffers)}"
          f" freed quantized/scale buffers & logits {logits_by_mode['bf16_ref'].shape} match offline BF16 bit-for-bit!"
      )


def verify_gemma4(repo_root: str, scan_layers: bool) -> None:
  """Verifies reduced-dim `gemma4-26b` shared `gate_up_proj` composite splitting vs `hf_streaming_load`."""
  base_yml = os.path.join(repo_root, "src/maxtext/configs/base.yml")
  with tempfile.TemporaryDirectory() as out_dir:
    config = pyconfig.initialize(
        [
            "",
            base_yml,
            "hardware=cpu",
            "skip_jax_distributed_system=True",
            f"base_output_directory={out_dir}",
            "model_name=gemma4-26b",
            "override_model_config=True",
            f"scan_layers={scan_layers}",
            "base_num_decoder_layers=6",
            "inhomogeneous_layer_cycle_interval=6",
            "base_emb_dim=64",
            "base_num_query_heads=4",
            "base_num_kv_heads=2",
            "global_num_kv_heads=2",
            "head_dim=16",
            "global_head_dim=16",
            "base_mlp_dim=64",
            "base_moe_mlp_dim=32",
            "num_experts=4",
            "num_experts_per_tok=2",
            "vocab_size=128",
            "weight_dtype=float32",
            "dtype=float32",
            "use_multimodal=False",
            "per_device_batch_size=1",
            "max_target_length=16",
        ]
    )
  model = model_creation_utils.from_config(config, rngs=nnx.Rngs(0))
  pure_weights = nnx.to_pure_dict(checkpointing._abstract_params(nnx.state(model)))  # pylint: disable=protected-access
  target_tree = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), pure_weights)

  hf_cfg = hf_model_configs.HF_MODEL_CONFIGS["gemma4-26b"].to_dict()
  hf_cfg["text_config"].update(
      {
          "hidden_size": 64,
          "intermediate_size": 64,
          "moe_intermediate_size": 32,
          "expert_intermediate_size": 32,
          "num_hidden_layers": 6,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "num_global_key_value_heads": 2,
          "head_dim": 16,
          "global_head_dim": 16,
          "num_experts": 4,
          "vocab_size": 128,
          "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
          "per_layer_config": None,
      }
  )
  hf_cfg.pop("vision_config", None)
  shapes = hf_shape.GEMMA4_HF_WEIGHTS_TO_SHAPE(hf_cfg)
  pm = param_mapping.PARAM_MAPPING["gemma4-26b"](hf_cfg, config, scan_layers=scan_layers)
  hm = param_mapping.HOOK_FNS["gemma4-26b"](hf_cfg, config, scan_layers=scan_layers, saving_to_hf=False)

  np_rng = np.random.default_rng(42)
  tensors = {
      k: jnp.asarray(np_rng.standard_normal(s, dtype=np.float32)).astype(jnp.bfloat16)
      for k, s in shapes.items()
      if isinstance(k, str)
  }
  with tempfile.TemporaryDirectory() as ckpt_dir:
    safetensors.flax.save_file(tensors, os.path.join(ckpt_dir, "model.safetensors"))
    hf_state = load_dynamic.rename_legacy_hf_keys(load_dynamic.load_sharded_hf_state(ckpt_dir), pm)
    restored = load_dynamic.transform_hf_state_to_mt_state(hf_state, target_tree, pm, hm, config)
    streamed = hf_streaming_load.load_hf_params_streaming(
        ckpt_dir,
        target_tree,
        pm,
        hm,
        config,
        rename_hf_keys=functools.partial(load_dynamic.rename_legacy_hf_keys, param_map_mt_to_hf=pm),
    )

  flat_r = flax.traverse_util.flatten_dict(restored["params"], sep=".")
  flat_s = flax.traverse_util.flatten_dict(streamed["params"], sep=".")
  assert set(flat_r) == set(flat_s)
  for k, arr in flat_r.items():
    assert isinstance(arr, jax.Array) and not arr.is_deleted(), f"{k} was deleted!"
    np.testing.assert_allclose(np.asarray(arr), np.asarray(flat_s[k]), rtol=0, atol=0, err_msg=k)
  print(f"  [PASS] gemma4-26b scan_layers={scan_layers}: {len(flat_r)} leaves verified, 0 deleted, matches streaming!")


def verify_glm52(repo_root: str, scan_layers: bool, prune_shared: bool) -> None:
  """Verifies reduced-dim `glm5.2-744b` `IndexShare` shared HF buffers vs `hf_streaming_load`."""
  base_yml = os.path.join(repo_root, "src/maxtext/configs/base.yml")
  with tempfile.TemporaryDirectory() as out_dir:
    config = pyconfig.initialize(
        [
            "",
            base_yml,
            "hardware=cpu",
            "skip_jax_distributed_system=True",
            f"base_output_directory={out_dir}",
            "model_name=glm5.2-744b",
            "override_model_config=True",
            f"scan_layers={scan_layers}",
            f"prune_shared_indexers={prune_shared}",
            "base_num_decoder_layers=5",
            "first_num_dense_layers=2",
            "index_share_pattern=FFSSF",
            "base_emb_dim=64",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=128",
            "base_moe_mlp_dim=32",
            "num_experts=4",
            "num_experts_per_tok=2",
            "vocab_size=128",
            "q_lora_rank=32",
            "kv_lora_rank=16",
            "qk_nope_head_dim=8",
            "qk_rope_head_dim=8",
            "v_head_dim=16",
            "indexer_n_heads=4",
            "indexer_head_dim=16",
            "indexer_topk=8",
            "weight_dtype=float32",
            "dtype=float32",
            "per_device_batch_size=1",
            "max_target_length=16",
        ]
    )
  model = model_creation_utils.from_config(config, rngs=nnx.Rngs(0))
  pure_weights = nnx.to_pure_dict(checkpointing._abstract_params(nnx.state(model)))  # pylint: disable=protected-access
  target_tree = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), pure_weights)

  hf_cfg = hf_model_configs.HF_MODEL_CONFIGS["glm5.2-744b"].to_dict()
  hf_cfg.pop("quantization_config", None)
  hf_cfg.update(
      {
          "hidden_size": 64,
          "intermediate_size": 128,
          "moe_intermediate_size": 32,
          "num_hidden_layers": 5,
          "first_k_dense_replace": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "n_routed_experts": 4,
          "num_experts_per_tok": 2,
          "n_shared_experts": 1,
          "vocab_size": 128,
          "q_lora_rank": 32,
          "kv_lora_rank": 16,
          "qk_nope_head_dim": 8,
          "qk_rope_head_dim": 8,
          "qk_head_dim": 16,
          "head_dim": 16,
          "v_head_dim": 16,
          "index_n_heads": 4,
          "index_head_dim": 16,
          "index_topk": 8,
          "indexer_types": ["full", "full", "shared", "shared", "full"],
          "index_share_pattern": "FFSSF",
          "num_nextn_predict_layers": 0,
      }
  )
  shapes = hf_shape.GLM5_2_HF_WEIGHTS_TO_SHAPE(hf_cfg)
  pm = param_mapping.PARAM_MAPPING["glm5.2-744b"](hf_cfg, config, scan_layers=scan_layers)
  hm = param_mapping.HOOK_FNS["glm5.2-744b"](hf_cfg, config, scan_layers=scan_layers, saving_to_hf=False)

  np_rng = np.random.default_rng(7)
  tensors = {
      k: jnp.asarray(np_rng.standard_normal(s, dtype=np.float32)).astype(jnp.bfloat16)
      for k, s in shapes.items()
      if isinstance(k, str)
  }
  with tempfile.TemporaryDirectory() as ckpt_dir:
    safetensors.flax.save_file(tensors, os.path.join(ckpt_dir, "model.safetensors"))
    hf_state = load_dynamic.rename_legacy_hf_keys(load_dynamic.load_sharded_hf_state(ckpt_dir), pm)
    restored = load_dynamic.transform_hf_state_to_mt_state(hf_state, target_tree, pm, hm, config)
    streamed = hf_streaming_load.load_hf_params_streaming(
        ckpt_dir,
        target_tree,
        pm,
        hm,
        config,
        rename_hf_keys=functools.partial(load_dynamic.rename_legacy_hf_keys, param_map_mt_to_hf=pm),
    )

  flat_r = flax.traverse_util.flatten_dict(restored["params"], sep=".")
  flat_s = flax.traverse_util.flatten_dict(streamed["params"], sep=".")
  assert set(flat_r) == set(flat_s)
  for k, arr in flat_r.items():
    assert isinstance(arr, jax.Array) and not arr.is_deleted(), f"{k} was deleted!"
    np.testing.assert_allclose(np.asarray(arr), np.asarray(flat_s[k]), rtol=0, atol=0, err_msg=k)
  print(
      f"  [PASS] glm5.2-744b scan_layers={scan_layers} prune_shared={prune_shared}:"
      f" {len(flat_r)} leaves verified, 0 deleted, matches streaming!"
  )


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint for local CPU end-to-end model verification."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo-root", default=_DEFAULT_REPO_ROOT, help="Path to the MaxText repository root.")
  parser.add_argument(
      "--models",
      nargs="+",
      choices=["deepseek4", "gemma4", "glm52"],
      default=["deepseek4", "gemma4", "glm52"],
      help="Subset of reduced-dim model suites to verify.",
  )
  args = parser.parse_args(argv)
  print(f"Running E2E CPU verification on {jax.device_count()} JAX devices (repo_root={args.repo_root})")
  if "deepseek4" in args.models:
    verify_deepseek4(args.repo_root)
  if "gemma4" in args.models:
    print("=== [2/3] Verifying reduced-dim gemma4-26b (shared composite gate_up_proj) ===")
    verify_gemma4(args.repo_root, scan_layers=True)
    verify_gemma4(args.repo_root, scan_layers=False)
  if "glm52" in args.models:
    print("=== [3/3] Verifying reduced-dim glm5.2-744b (IndexShare zero-copy buffer pinning) ===")
    verify_glm52(args.repo_root, scan_layers=True, prune_shared=False)
    verify_glm52(args.repo_root, scan_layers=False, prune_shared=True)
    verify_glm52(args.repo_root, scan_layers=False, prune_shared=False)
  print("[PASS] All local CPU E2E model parity & buffer-hygiene checks succeeded.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
