"""
Copyright 2023 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

r"""Convert weights from a Llama for MaxText inference.

Usage:

Get LLaMA pytorch_vars from Meta

Example cmd:
To save a ckpt
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path <path/to/meta/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size llama2-7b

For large size model (e.g. 70B model), this script requires large memory VM.
The script load and save weights in a single pass.
To fit less memory, modify convert() to load/save weights in multiple passes.
Each pass, load and save partial weights (subset of all weight variables).
"""
# pylint: disable=g-line-too-long
import argparse
import pathlib
import dataclasses
import collections
from collections.abc import Callable, Generator, MutableMapping, Sequence

import numpy as np

import checkpointing
import jax
from flax.training import train_state
import max_logging
import max_utils
from train import save_checkpoint
# import torch
import sys
import os

jax.config.update("jax_platform_name", "cpu")


def permute_to_match_maxtext_rope(arr):
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)


MODEL_PARAMS_DICT = {
    "llama2-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 32000,
    },
    "llama2-13b": {
        "num_layers": 40,
        "num_heads": 40,
        "num_kv_heads": 40,
        "dims_per_head": 128,
        "vocab": 32000,
    },
    "llama2-7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "dims_per_head": 128,
        "vocab": 32000,
    },
    "llama3-8b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3-405b": {
        "num_layers": 126,
        "num_heads": 128,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
}


def convert(base_model_path, maxtext_model_path, model_size, mesh):
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and save at maxtext_model_path

  Attributes:
  base_model_path: checkpoint path
  maxtext_model_path: Path to save the MaxText checkpoint to
  model_size: llama3-8b to 405b.
  """
  """Convert model to maxtext."""
  model_params = MODEL_PARAMS_DICT[model_size]
  base_num_decoder_layers = model_params["num_layers"]
  base_num_query_heads = model_params["num_heads"]
  head_dim = model_params["dims_per_head"]
  base_num_kv_heads = model_params["num_kv_heads"]
  vocab_size = model_params["vocab"]

  print(f"Loading the base model from {base_model_path}")
  # Skip any hidden files for checkpoints
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.pth"))
  pytorch_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    print(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    import psutil
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    pytorch_vars[int(ckpt_path.name.split(".", maxsplit=2)[1])] = checkpoint
    print("memory usage in GB: ", psutil.Process().memory_info().rss / (1024 * 1024))

  pytorch_vars = [pytorch_vars[i] for i in sorted(list(pytorch_vars.keys()))]

  if model_size[:6] == 'llama3':
    token_embedder = np.concatenate(
      [var["tok_embeddings.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
    )
  else:
    token_embedder = np.concatenate(
      [var["tok_embeddings.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=1
    )[:vocab_size, :]

  for var in pytorch_vars:
    del var["tok_embeddings.weight"]
  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": pytorch_vars[0]["norm.weight"].type(torch.float16).numpy()},
          "logits_dense": {
              "kernel": np.concatenate(
                  [var["output.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
              ).transpose()[:, :vocab_size]
          },
      },
      "token_embedder": {
          "embedding": token_embedder
      },
  }
  for i in range(base_num_decoder_layers):
    jax_weights["decoder"][f"layers_{i}"] = {
      "mlp": {
        "wi_0": {"kernel": None},
        "wi_1": {"kernel": None},
        "wo": {"kernel": None},
      },
      "pre_self_attention_layer_norm": {"scale": None},
      "post_self_attention_layer_norm": {"scale": None},
      "self_attention": {
        "query": {"kernel": None},
        "key": {"kernel": None},
        "value": {"kernel": None},
        "out": {"kernel": None},
      },
    }

  # llama3-405b kv weight is replicated within every two files.
  wkv_step = 1 if model_size != "llama3-405b" else 2

  for layer_idx in range(base_num_decoder_layers):
    print("layer idx: ", layer_idx)
    print("memory usage in GB: ", psutil.Process().memory_info().rss / (1024 * 1024))
    wq = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wq.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
    ).transpose()
    wk = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wk.weight"].type(torch.float16).numpy() for var in pytorch_vars[::wkv_step]], axis=0
    ).transpose()
    wv = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wv.weight"].type(torch.float16).numpy() for var in pytorch_vars[::wkv_step]], axis=0
    ).transpose()

    wq = np.reshape(wq, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])
    wk = np.reshape(wk, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
    wv = np.reshape(wv, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
    wq = permute_to_match_maxtext_rope(wq)
    wk = permute_to_match_maxtext_rope(wk)

    w_post = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wo.weight"].type(torch.float16).numpy() for var in pytorch_vars],
        axis=1,
    )

    w_post = np.reshape(w_post, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])

    jax_weights["decoder"][f"layers_{layer_idx}"]["self_attention"]["query"]["kernel"] = wq / np.sqrt(head_dim)

    jax_weights["decoder"][f"layers_{layer_idx}"]["self_attention"]["key"]["kernel"] = wk
    jax_weights["decoder"][f"layers_{layer_idx}"]["self_attention"]["value"]["kernel"] = wv

    # base_num_query_heads * head_dim, base_num_query_heads, head_dim =>
    # base_num_query_heads, head_dim, base_num_query_heads * head_dim
    jax_weights["decoder"][f"layers_{layer_idx}"]["self_attention"]["out"]["kernel"] = np.transpose(w_post, axes=(1,2,0))

    pre_self_attention_layernorm = pytorch_vars[0][f"layers.{layer_idx}.attention_norm.weight"].type(torch.float16).numpy()
    post_self_attention_layernorm = pytorch_vars[0][f"layers.{layer_idx}.ffn_norm.weight"].type(torch.float16).numpy()
    jax_weights["decoder"][f"layers_{layer_idx}"]["pre_self_attention_layer_norm"]["scale"] = pre_self_attention_layernorm
    jax_weights["decoder"][f"layers_{layer_idx}"]["post_self_attention_layer_norm"]["scale"] = post_self_attention_layernorm


    wi_0 = np.concatenate(
        [var[f"layers.{layer_idx}.feed_forward.w1.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
    ).transpose()
    wi_1 = np.concatenate(
        [var[f"layers.{layer_idx}.feed_forward.w3.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
    ).transpose()
    wo = np.concatenate(
        [var[f"layers.{layer_idx}.feed_forward.w2.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=1
    ).transpose()
    jax_weights["decoder"][f"layers_{layer_idx}"]["mlp"]["wi_0"]["kernel"] = wi_0
    jax_weights["decoder"][f"layers_{layer_idx}"]["mlp"]["wi_1"]["kernel"] = wi_1
    jax_weights["decoder"][f"layers_{layer_idx}"]["mlp"]["wo"]["kernel"] = wo

    for var in pytorch_vars:
      del var[f"layers.{layer_idx}.attention.wq.weight"]
      del var[f"layers.{layer_idx}.attention.wk.weight"]
      del var[f"layers.{layer_idx}.attention.wv.weight"]
      del var[f"layers.{layer_idx}.attention.wo.weight"]
      del var[f"layers.{layer_idx}.feed_forward.w1.weight"]
      del var[f"layers.{layer_idx}.feed_forward.w2.weight"]
      del var[f"layers.{layer_idx}.feed_forward.w3.weight"]

  s1 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("checkpoint_sharding_axis"))  # shards first axis
  s2 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "checkpoint_sharding_axis"))  # shards second axis
  s3 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))  # no sharding

  def checkpoint_device_put(arr):
    if arr.shape[0] % SIMULATED_CPU_DEVICES_COUNT == 0:
      print("sharding first axis")
      return jax.device_put(arr, device=s1)
    elif len(arr.shape) > 1 and arr.shape[1] % SIMULATED_CPU_DEVICES_COUNT == 0:
      print("sharding second axis")
      return jax.device_put(arr, device=s2)
    else:
      print("no sharding was possible, replicating")
      return jax.device_put(arr, device=s3)

  # convert all weights to jax.numpy with sharding if applicable
  jax_weights = jax.tree_util.tree_map(checkpoint_device_put, jax_weights)

  # dummy configs for the checkpoint_manager
  step_number_to_save_new_ckpt = 0
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      checkpoint_dir=maxtext_model_path,
      enable_checkpointing=True,
      use_async=False,
      save_interval_steps=1,
      use_ocdbt=False,
      use_zarr3=False,
  )

  state_new = train_state.TrainState(
      step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={}  # type: ignore
  )

  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, step_number_to_save_new_ckpt, state_new):
      max_logging.log(f"saved a checkpoint at step {step_number_to_save_new_ckpt}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()


# Needs to support fields read in https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/max_utils.py#L363
class ParallelismConfig:
  def __init__(self, args):
    self.allow_split_physical_axes = False

    self.num_slices = args.num_slices
    self.dcn_data_parallelism = args.dcn_data_parallelism
    self.dcn_pipeline_parallelism = args.dcn_pipeline_parallelism
    self.dcn_fsdp_parallelism = args.dcn_fsdp_parallelism
    self.dcn_fsdp_transpose_parallelism = args.dcn_fsdp_transpose_parallelism
    self.dcn_sequence_parallelism = args.dcn_sequence_parallelism
    self.dcn_tensor_parallelism = args.dcn_tensor_parallelism
    self.dcn_expert_parallelism = args.dcn_expert_parallelism
    self.dcn_autoregressive_parallelism = args.dcn_autoregressive_parallelism
    self.ici_data_parallelism = args.ici_data_parallelism
    self.ici_pipeline_parallelism = args.ici_pipeline_parallelism
    self.ici_fsdp_parallelism = args.ici_fsdp_parallelism
    self.ici_fsdp_transpose_parallelism = args.ici_fsdp_transpose_parallelism
    self.ici_sequence_parallelism = args.ici_sequence_parallelism
    self.ici_tensor_parallelism = args.ici_tensor_parallelism
    self.ici_expert_parallelism = args.ici_expert_parallelism
    self.ici_autoregressive_parallelism = args.ici_autoregressive_parallelism


# START copy from https://github.com/jax-ml/jax/blob/main/tests/mesh_utils_test.py
@dataclasses.dataclass(frozen=True)
class MockClient:
  """Mock client for testing, everything is done as process index 0."""
  def process_index(self) -> int:
    return 0


@dataclasses.dataclass(frozen=True)
class MockTpuDevice:
  """Mock TPU device for testing."""
  id: int
  platform: str
  device_kind: str
  process_index: int
  coords: Sequence[int]
  core_on_chip: int
  slice_index: int = 0
  client: MockClient = dataclasses.field(default_factory=MockClient)


def mock_tpu_devices(x, y, z, dev_kind, one_device_per_chip, num_slices=1,
                     reorder=False):
  """Produce fake jax.devices() output for a TPU slice."""
  assert x > 0 and y > 0 and z > 0

  cores_per_chip = 1 if one_device_per_chip else 2

  # 3D shape of the mesh of devices on each host (= process).
  nxd, nyd, nzd = (min(x, 2), min(y, 2), 1)
  # 3D shape of the mesh of hosts (= processes):
  nxp, nyp, nzp = x // nxd, y // nyd, z // nzd
  assert nxp * nxd == x
  assert nyp * nyd == y
  assert nzp * nzd == z

  def mock_tpu_device(core_on_chip, xd, yd, zd, xp, yp, zp, slice_index):
    process_index = xp + nxp * (yp + nyp * (zp + nzp * slice_index))
    coords =  (xd + nxd * xp, yd + nyd * yp, zd + nzd * zp)
    device_id = core_on_chip + cores_per_chip * (xd + nxd * (xp + nxp * (
        yd + nyd * (yp + nyp * (zd + nzd * (zp + nzp * slice_index))))))
    return MockTpuDevice(device_id, 'tpu', dev_kind, process_index, coords,
                         core_on_chip, slice_index)
  devices = [mock_tpu_device(core_on_chip, xd, yd, zd, xp, yp, zp, slice_index)
             for slice_index in range(num_slices)
             for zp in range(nzp) for yp in range(nyp) for xp in range(nxp)
             for zd in range(nzd) for yd in range(nyd) for xd in range(nxd)
             for core_on_chip in range(cores_per_chip)]
  if reorder:
    devices = devices[::-1]

  # Validate the generated mock devices:
  num_local_chips = nxd * nyd  # Number of mock devices / process.
  if num_local_chips < 4:
    # Sub-host slice = fewer than the 4 chips available on a host:
    # e.g., 1x1 TPU v2.  All devices should be on one host.
    num_all_chips = x * y * z
    assert num_all_chips == num_local_chips, f'Bad shape: {x=}, {y=}, {z=}'
    # Implied by the previous assertion, but let's be explicit:
    assert z == 1
    _validate_mocked_devices_for_subhost_slice(devices, x, y, cores_per_chip)
  else:
    _validate_mocked_devices(devices, num_local_chips * cores_per_chip)

  return devices


# If this function raises, it's a bug in the test code!
def _validate_mocked_devices_for_subhost_slice(devices, x, y, cores_per_chip):
  first_device = devices[0]
  distinct_coords = set()
  for d in devices:
    assert d.process_index == first_device.process_index
    assert d.coords[0] >= 0 and d.coords[0] < x
    assert d.coords[1] >= 0 and d.coords[1] < y
    assert d.coords[2] == 0
    assert d.core_on_chip >= 0 and d.core_on_chip < cores_per_chip
    distinct_coords.add((d.coords[0], d.coords[1], 0, d.core_on_chip))
  assert len(distinct_coords) == x * y * cores_per_chip


# If this function raises, it's a bug in the test code!
def _validate_mocked_devices(devices, num_local_devices):
  # NOTE: this function is not called for sub-host slices.
  process_to_devices = collections.defaultdict(list)
  for d in devices:
    process_to_devices[d.process_index].append(d)

  for local_devices in process_to_devices.values():
    assert len(local_devices) == num_local_devices, local_devices
    # All devices have same z coord
    assert len({d.coords[2] for d in local_devices}) == 1, local_devices
    # All devices in a 2x2 subgrid
    min_coords = min(d.coords for d in local_devices)
    expected = set()
    for x, y in [(0,0), (0,1), (1,0), (1,1)]:
      expected.add((min_coords[0] + x, min_coords[1] + y, min_coords[2]))
    assert {d.coords for d in local_devices} == expected, local_devices
# END copy from https://github.com/jax-ml/jax/blob/main/tests/mesh_utils_test.py


def topology(t):
  x, y, z = t.split("x")
  return int(x), int(y), int(z)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-model-path", type=str, required=True)
  parser.add_argument("--maxtext-model-path", type=str, required=True)
  parser.add_argument("--model-size", type=str, required=True, choices=MODEL_PARAMS_DICT.keys())
  parser.add_argument("--tpu-device-kind", type=str, required=True, choices=['TPU v2', 'TPU v3', 'TPU v4', 'TPU v5 lite', 'TPU v5'])
  parser.add_argument("--topology", type=str, required=True)
  parser.add_argument("--num-slices", type=int, default=1)
  parser.add_argument("--dcn-data-parallelism", type=int, default=1)
  parser.add_argument("--dcn-pipeline-parallelism", type=int, default=1)
  parser.add_argument("--dcn-fsdp-parallelism", type=int, default=1)
  parser.add_argument("--dcn-fsdp-transpose-parallelism", type=int, default=1)
  parser.add_argument("--dcn-sequence-parallelism", type=int, default=1)
  parser.add_argument("--dcn-tensor-parallelism", type=int, default=1)
  parser.add_argument("--dcn-expert-parallelism", type=int, default=1)
  parser.add_argument("--dcn-autoregressive-parallelism", type=int, default=1)
  parser.add_argument("--ici-data-parallelism", type=int, default=1)
  parser.add_argument("--ici-pipeline-parallelism", type=int, default=1)
  parser.add_argument("--ici-fsdp-parallelism", type=int, default=1)
  parser.add_argument("--ici-fsdp-transpose-parallelism", type=int, default=1)
  parser.add_argument("--ici-sequence-parallelism", type=int, default=1)
  parser.add_argument("--ici-tensor-parallelism", type=int, default=1)
  parser.add_argument("--ici-expert-parallelism", type=int, default=1)
  parser.add_argument("--ici-autoregressive-parallelism", type=int, default=1)

  args = parser.parse_args()

  config = ParallelismConfig(args)
  x, y, z = topology(args.topology)
  devices = mock_tpu_devices(x, y, z, dev_kind=args.tpu_device_kind, one_device_per_chip=True, num_slices=args.num_slices)
  mesh = max_utils.create_device_mesh(config, devices)

  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={len(devices)}"

  convert(args.base_model_path, args.maxtext_model_path, args.model_size, mesh)
