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

from __future__ import annotations

import json
import math
import os
import struct
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

from maxtext.experimental.flint40b import flint_checkpoint
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b.tests import hf_reference


def _raw_bits_view(arr: np.ndarray) -> np.ndarray:
  """Returns an unsigned-integer bit view for exact IEEE/BF16 comparison."""
  contig = np.ascontiguousarray(arr)
  if contig.dtype == np.dtype(ml_dtypes.bfloat16) or contig.dtype == np.dtype(np.float16):
    return contig.view(np.uint16)
  if contig.dtype == np.dtype(np.float32):
    return contig.view(np.uint32)
  raise AssertionError(f"Unexpected array dtype in test: {contig.dtype!r}.")


def _write_raw_safetensors(
    file_path: str,
    header_obj: dict[str, object],
    payload: bytes,
) -> None:
  """Writes a raw `.safetensors` file with arbitrary header and payload."""
  header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
  with open(file_path, "wb") as out_f:
    out_f.write(struct.pack("<Q", len(header_bytes)))
    out_f.write(header_bytes)
    out_f.write(payload)


class FlintCheckpointTest(parameterized.TestCase):

  def test_40b_exact_key_count_and_literal_hf_key_names(self):
    model_40b = flint_config.flint_40b_config().model
    model_mini = flint_config.mini_flint_config().model

    shapes_40b = flint_checkpoint.expected_hf_shapes(model_40b)
    shapes_mini = flint_checkpoint.expected_hf_shapes(model_mini)

    self.assertLen(shapes_40b, 36_734)
    self.assertLen(shapes_mini, 454)
    self.assertEqual(set(shapes_40b), hf_reference.expected_hf_state_dict_keys(model_40b))
    self.assertEqual(set(shapes_mini), hf_reference.expected_hf_state_dict_keys(model_mini))

    total_elems = sum(math.prod(shape) for shape in shapes_40b.values())
    self.assertEqual(total_elems, 39_785_287_456)

    self.assertEqual(shapes_40b["model.embed_tokens.weight"], (150144, 2048))
    self.assertEqual(shapes_40b["lm_head.weight"], (150144, 2048))
    self.assertEqual(shapes_40b["model.norm.weight"], (2048,))
    self.assertEqual(shapes_40b["model.layers.0.self_attn.softmax_offset"], (32,))
    self.assertEqual(shapes_40b["model.layers.0.self_attn.q_proj.weight"], (4096, 2048))
    self.assertEqual(shapes_40b["model.layers.0.self_attn.k_proj.weight"], (512, 2048))
    self.assertEqual(shapes_40b["model.layers.0.self_attn.v_proj.weight"], (512, 2048))
    self.assertEqual(shapes_40b["model.layers.0.self_attn.o_proj.weight"], (2048, 4096))
    self.assertEqual(shapes_40b["model.layers.0.self_attn.q_norm.weight"], (128,))
    self.assertEqual(shapes_40b["model.layers.0.self_attn.k_norm.weight"], (128,))
    self.assertEqual(shapes_40b["model.layers.0.mlp.gate_proj.weight"], (6144, 2048))
    self.assertEqual(shapes_40b["model.layers.0.mlp.up_proj.weight"], (6144, 2048))
    self.assertEqual(shapes_40b["model.layers.0.mlp.down_proj.weight"], (2048, 6144))
    self.assertEqual(shapes_40b["model.layers.2.mlp.gate.weight"], (256, 2048))
    self.assertEqual(
        shapes_40b["model.layers.2.mlp.shared_expert.gate_proj.weight"],
        (1024, 2048),
    )
    self.assertEqual(
        shapes_40b["model.layers.2.mlp.shared_expert.up_proj.weight"],
        (1024, 2048),
    )
    self.assertEqual(
        shapes_40b["model.layers.2.mlp.shared_expert.down_proj.weight"],
        (2048, 1024),
    )
    self.assertEqual(
        shapes_40b["model.layers.48.mlp.experts.255.down_proj.weight"],
        (2048, 512),
    )

    for key in shapes_40b:
      self.assertNotIn("shared_experts", key)
    self.assertNotIn("model.layers.2.mlp.shared_experts.gate_proj.weight", shapes_40b)

  @parameterized.named_parameters(
      dict(testcase_name="f32", hf_dtype="F32"),
      dict(testcase_name="bf16", hf_dtype="BF16"),
  )
  def test_f32_and_bf16_multishard_roundtrip_scanned_and_unscanned_sharded_mesh(self, hf_dtype: str):
    base_cfg = flint_config.mini_flint_config(
        scan_layers=False,
        fsdp_size=1,
        cp_size=1,
        ep_size=1,
        global_batch_size=8,
        param_dtype="float32",
    )
    torch_model = hf_reference.TorchFlintForCausalLM(base_cfg.model)
    hf_reference.randomize_torch_flint(torch_model, seed=29)
    raw_sd = torch_model.state_dict()

    mesh_1 = flint_sharding.create_mesh(base_cfg.sharding, devices=jax.devices()[:1])
    src_params = hf_reference.torch_state_dict_to_flint_params(raw_sd, base_cfg, mesh_1)

    ckpt_dir = os.path.join(self.create_tempdir().full_path, f"multishard_{hf_dtype.lower()}")
    shard_paths = flint_checkpoint.save_hf_checkpoint(
        src_params,
        base_cfg,
        ckpt_dir,
        hf_dtype=hf_dtype,
        max_shard_bytes=65536,
    )
    self.assertGreater(len(shard_paths), 1)
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    self.assertTrue(os.path.exists(index_path))

    orig_reader = flint_checkpoint.SafetensorsReader(ckpt_dir)
    self.assertLen(orig_reader.keys(), 454)
    meta_shape, meta_dtype = orig_reader.tensor_meta("model.layers.0.self_attn.softmax_offset")
    self.assertEqual(meta_shape, (base_cfg.model.num_attention_heads,))
    self.assertEqual(meta_dtype, hf_dtype)

    if hf_dtype == "BF16":
      ref_sd = {k: v.to(torch.bfloat16) for k, v in raw_sd.items()}
    else:
      ref_sd = raw_sd

    mesh_shapes = ((1, 1, 1), (4, 2, 1), (2, 1, 4), (2, 2, 2))
    loaded_222_scanned = None
    cfg_222_scanned = None

    def _make_read_spy():
      orig_read_fn = flint_checkpoint.SafetensorsReader.read_tensor
      calls: list[str] = []

      def _spy_read(reader_self, key: str, *, dtype=None):
        calls.append(key)
        return orig_read_fn(reader_self, key, dtype=dtype)

      return calls, _spy_read

    for scan_layers in (True, False):
      for fsdp_size, cp_size, ep_size in mesh_shapes:
        cfg = flint_config.mini_flint_config(
            scan_layers=scan_layers,
            fsdp_size=fsdp_size,
            cp_size=cp_size,
            ep_size=ep_size,
            global_batch_size=8,
            param_dtype="float32",
        )
        devs = jax.devices()[: cfg.sharding.num_devices]
        mesh = flint_sharding.create_mesh(cfg.sharding, devices=devs)
        expected_shardings = flint_sharding.named_param_shardings(mesh, cfg)
        ref_params = hf_reference.torch_state_dict_to_flint_params(ref_sd, cfg, mesh)

        if scan_layers and (fsdp_size, cp_size, ep_size) == (2, 2, 2):
          read_calls, spy_read_fn = _make_read_spy()
          with mock.patch.object(
              flint_checkpoint.SafetensorsReader,
              "read_tensor",
              autospec=True,
              side_effect=spy_read_fn,
          ):
            got_params = flint_checkpoint.load_hf_checkpoint(ckpt_dir, cfg, mesh)
          # 8 local devices must read each HF tensor strictly once (454 total).
          self.assertLen(read_calls, 454)
          loaded_222_scanned = got_params
          cfg_222_scanned = cfg
        else:
          got_params = flint_checkpoint.load_hf_checkpoint(ckpt_dir, cfg, mesh)

        self.assertEqual(
            got_params.decoder.prefix_dense[0].self_attn.sinks.dtype,
            jnp.float32,
        )
        if scan_layers:
          sb = got_params.decoder.scanned_blocks
          self.assertIsNotNone(sb)
          assert sb is not None
          self.assertEqual(sb.sliding_0.self_attn.sinks.dtype, jnp.float32)
          self.assertEqual(sb.global_3.self_attn.sinks.dtype, jnp.float32)
          self.assertEqual(got_params.decoder.tail_moe[0].self_attn.sinks.dtype, jnp.float32)
        else:
          self.assertEqual(
              got_params.decoder.unscanned_moe[0].self_attn.sinks.dtype,
              jnp.float32,
          )

        for got_leaf, ref_leaf, exp_sh in zip(
            jax.tree.leaves(got_params),
            jax.tree.leaves(ref_params),
            jax.tree.leaves(expected_shardings),
            strict=True,
        ):
          self.assertEqual(got_leaf.sharding, exp_sh)
          self.assertEqual(got_leaf.dtype, ref_leaf.dtype)
          np.testing.assert_array_equal(
              _raw_bits_view(np.asarray(got_leaf)),
              _raw_bits_view(np.asarray(ref_leaf)),
          )

    # Also verify param_dtype="bfloat16" keeps sinks in float32.
    cfg_bf16_param = flint_config.mini_flint_config(
        scan_layers=True,
        fsdp_size=2,
        cp_size=2,
        ep_size=2,
        global_batch_size=8,
        param_dtype="bfloat16",
    )
    mesh_8 = flint_sharding.create_mesh(cfg_bf16_param.sharding)
    got_bf16_param = flint_checkpoint.load_hf_checkpoint(ckpt_dir, cfg_bf16_param, mesh_8)
    self.assertEqual(got_bf16_param.embed.dtype, jnp.bfloat16)
    self.assertEqual(
        got_bf16_param.decoder.prefix_dense[0].self_attn.sinks.dtype,
        jnp.float32,
    )
    assert got_bf16_param.decoder.scanned_blocks is not None
    self.assertEqual(
        got_bf16_param.decoder.scanned_blocks.sliding_0.self_attn.sinks.dtype,
        jnp.float32,
    )

    # Re-export from scan_layers=True sharded (2, 2, 2) and compare bit-for-bit.
    self.assertIsNotNone(loaded_222_scanned)
    self.assertIsNotNone(cfg_222_scanned)
    assert loaded_222_scanned is not None
    assert cfg_222_scanned is not None

    reexport_dir = os.path.join(self.create_tempdir().full_path, f"reexport_{hf_dtype.lower()}")
    reexport_paths = flint_checkpoint.save_hf_checkpoint(
        loaded_222_scanned,
        cfg_222_scanned,
        reexport_dir,
        hf_dtype=hf_dtype,
        max_shard_bytes=65536,
    )
    self.assertLen(reexport_paths, len(shard_paths))

    reexport_reader = flint_checkpoint.SafetensorsReader(reexport_dir)
    self.assertEqual(reexport_reader.keys(), orig_reader.keys())
    for key in sorted(orig_reader.keys()):
      self.assertEqual(reexport_reader.tensor_meta(key), orig_reader.tensor_meta(key))
      orig_arr = orig_reader.read_tensor(key)
      reexp_arr = reexport_reader.read_tensor(key)
      np.testing.assert_array_equal(_raw_bits_view(reexp_arr), _raw_bits_view(orig_arr))

    for orig_shard, reexp_shard in zip(shard_paths, reexport_paths, strict=True):
      with (
          open(orig_shard, "rb") as f_orig,
          open(reexp_shard, "rb") as f_reexp,
      ):
        self.assertEqual(f_reexp.read(), f_orig.read())

  def test_single_file_safetensors_and_metadata_ignored(self):
    cfg = flint_config.mini_flint_config(
        scan_layers=True,
        fsdp_size=2,
        cp_size=2,
        ep_size=2,
        global_batch_size=8,
        param_dtype="float32",
    )
    mesh = flint_sharding.create_mesh(cfg.sharding)
    torch_model = hf_reference.TorchFlintForCausalLM(cfg.model)
    hf_reference.randomize_torch_flint(torch_model, seed=77)
    ref_params = hf_reference.torch_state_dict_to_flint_params(torch_model.state_dict(), cfg, mesh)

    single_dir = os.path.join(self.create_tempdir().full_path, "single_ckpt")
    written = flint_checkpoint.save_hf_checkpoint(ref_params, cfg, single_dir, hf_dtype="F32")
    self.assertEqual(written, [os.path.join(single_dir, "model.safetensors")])
    self.assertFalse(os.path.exists(os.path.join(single_dir, "model.safetensors.index.json")))

    # Inject custom __metadata__ fields and verify both dir and file paths load.
    single_file = written[0]
    with open(single_file, "rb") as in_f:
      (hdr_len,) = struct.unpack("<Q", in_f.read(8))
      hdr_obj = json.loads(in_f.read(hdr_len).decode("utf-8"))
      payload = in_f.read()
    self.assertIn("__metadata__", hdr_obj)
    hdr_obj["__metadata__"] = {"format": "pt", "framework": "flint_unit_test"}
    _write_raw_safetensors(single_file, hdr_obj, payload)

    reader_dir = flint_checkpoint.SafetensorsReader(single_dir)
    reader_file = flint_checkpoint.SafetensorsReader(single_file)
    self.assertNotIn("__metadata__", reader_dir.keys())
    self.assertEqual(reader_dir.keys(), reader_file.keys())
    self.assertLen(reader_dir.keys(), 454)

    loaded_from_dir = flint_checkpoint.load_hf_checkpoint(single_dir, cfg, mesh)
    loaded_from_file = flint_checkpoint.load_hf_checkpoint(single_file, cfg, mesh)
    for a_dir, a_file, a_ref in zip(
        jax.tree.leaves(loaded_from_dir),
        jax.tree.leaves(loaded_from_file),
        jax.tree.leaves(ref_params),
        strict=True,
    ):
      np.testing.assert_array_equal(np.asarray(a_dir), np.asarray(a_ref))
      np.testing.assert_array_equal(np.asarray(a_file), np.asarray(a_ref))

    # Verify F16 safetensors tensor decoding as well.
    f16_path = os.path.join(self.create_tempdir().full_path, "f16.safetensors")
    f16_vals = np.array([[1.5, -2.25], [0.0, 4.0]], dtype=np.dtype("<f2"))
    _write_raw_safetensors(
        f16_path,
        {
            "__metadata__": {"format": "pt"},
            "toy.weight": {
                "dtype": "F16",
                "shape": [2, 2],
                "data_offsets": [0, f16_vals.nbytes],
            },
        },
        f16_vals.tobytes(),
    )
    f16_reader = flint_checkpoint.SafetensorsReader(f16_path)
    np.testing.assert_array_equal(f16_reader.read_tensor("toy.weight"), f16_vals)
    np.testing.assert_array_equal(
        f16_reader.read_tensor("toy.weight", dtype=np.float32),
        f16_vals.astype(np.float32),
    )

  def test_validation_fails_fast_on_missing_extra_bad_shape_or_dtype(self):
    cfg = flint_config.mini_flint_config(
        scan_layers=True,
        fsdp_size=1,
        cp_size=1,
        ep_size=1,
        global_batch_size=8,
    )
    mesh = flint_sharding.create_mesh(cfg.sharding, devices=jax.devices()[:1])
    params = flint_sharding.init_random_params(jax.random.PRNGKey(3), mesh, cfg)

    valid_dir = os.path.join(self.create_tempdir().full_path, "valid")
    valid_file = flint_checkpoint.save_hf_checkpoint(params, cfg, valid_dir, hf_dtype="BF16")[0]
    with open(valid_file, "rb") as in_f:
      (hdr_len,) = struct.unpack("<Q", in_f.read(8))
      base_hdr = json.loads(in_f.read(hdr_len).decode("utf-8"))
      base_payload = in_f.read()

    # 1. Missing key ("lm_head.weight").
    missing_path = os.path.join(self.create_tempdir().full_path, "missing.safetensors")
    missing_hdr = dict(base_hdr)
    del missing_hdr["lm_head.weight"]
    _write_raw_safetensors(missing_path, missing_hdr, base_payload)
    with self.assertRaisesRegex(ValueError, r"missing.*lm_head\.weight"):
      flint_checkpoint.load_hf_checkpoint(missing_path, cfg, mesh)

    # 2. Unexpected key (plural "shared_experts").
    extra_path = os.path.join(self.create_tempdir().full_path, "extra.safetensors")
    extra_hdr = dict(base_hdr)
    extra_end = len(base_payload)
    extra_hdr["model.layers.2.mlp.shared_experts.gate_proj.weight"] = {
        "dtype": "BF16",
        "shape": [2, 2],
        "data_offsets": [extra_end, extra_end + 8],
    }
    _write_raw_safetensors(extra_path, extra_hdr, base_payload + (b"\x00" * 8))
    with self.assertRaisesRegex(ValueError, r"unexpected.*shared_experts"):
      flint_checkpoint.load_hf_checkpoint(extra_path, cfg, mesh)

    # 3. Wrong tensor shape ("model.norm.weight": [2, 64] vs [128], same bytes).
    bad_shape_path = os.path.join(self.create_tempdir().full_path, "bad_shape.safetensors")
    bad_shape_hdr = json.loads(json.dumps(base_hdr))
    bad_shape_hdr["model.norm.weight"]["shape"] = [2, 64]
    _write_raw_safetensors(bad_shape_path, bad_shape_hdr, base_payload)
    with self.assertRaisesRegex(ValueError, r"shape mismatch.*model\.norm"):
      flint_checkpoint.load_hf_checkpoint(bad_shape_path, cfg, mesh)

    # 4. Unsupported dtype "I64".
    bad_dtype_path = os.path.join(self.create_tempdir().full_path, "bad_dtype.safetensors")
    _write_raw_safetensors(
        bad_dtype_path,
        {
            "bad.tensor": {
                "dtype": "I64",
                "shape": [1],
                "data_offsets": [0, 8],
            }
        },
        b"\x00" * 8,
    )
    with self.assertRaisesRegex(ValueError, r"Unsupported safetensors dtype 'I64'"):
      flint_checkpoint.SafetensorsReader(bad_dtype_path)

    # 5. Negative / misordered offsets and truncated byte payload.
    misordered_path = os.path.join(self.create_tempdir().full_path, "misordered.safetensors")
    _write_raw_safetensors(
        misordered_path,
        {
            "bad.offsets": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [8, 4],
            }
        },
        b"\x00" * 8,
    )
    with self.assertRaisesRegex(ValueError, "Negative or misordered"):
      flint_checkpoint.SafetensorsReader(misordered_path)

    truncated_path = os.path.join(self.create_tempdir().full_path, "truncated.safetensors")
    _write_raw_safetensors(
        truncated_path,
        {
            "trunc.tensor": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, 16],
            }
        },
        b"\x00" * 4,
    )
    with self.assertRaisesRegex(ValueError, "Truncated byte payload"):
      flint_checkpoint.SafetensorsReader(truncated_path)

    overflow_path = os.path.join(self.create_tempdir().full_path, "overflow.safetensors")
    with open(overflow_path, "wb") as ov_f:
      ov_f.write(struct.pack("<Q", 1024))
      ov_f.write(b"{}")
    with self.assertRaisesRegex(ValueError, "header overflow"):
      flint_checkpoint.SafetensorsReader(overflow_path)

    bad_idx_dir = os.path.join(self.create_tempdir().full_path, "bad_idx")
    os.makedirs(bad_idx_dir, exist_ok=True)
    shard_a = os.path.join(bad_idx_dir, "model-00001-of-00001.safetensors")
    _write_raw_safetensors(
        shard_a,
        {
            "actual.key": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        },
        b"\x00" * 4,
    )
    with open(os.path.join(bad_idx_dir, "model.safetensors.index.json"), "wb") as idx_f:
      idx_f.write(json.dumps({"weight_map": {"wrong.key": "model-00001-of-00001.safetensors"}}).encode("utf-8"))
    with self.assertRaisesRegex(ValueError, "weight_map disagrees with shard headers"):
      flint_checkpoint.SafetensorsReader(bad_idx_dir)

    # 6. Missing directory raises FileNotFoundError.
    with self.assertRaises(FileNotFoundError):
      flint_checkpoint.SafetensorsReader(os.path.join(self.create_tempdir().full_path, "does_not_exist"))


if __name__ == "__main__":
  absltest.main()
