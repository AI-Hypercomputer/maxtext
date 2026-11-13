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

"""Ahead-of-Time (AOT) compilation, HBM budget, and HLO vision-path checker for DeepSeek-V4.1-Flash."""

import argparse
from collections.abc import Iterator, Mapping, Sequence
import contextlib
import copy
import json
import os
import re
import subprocess
import sys
from typing import Any

from flax import nnx
from flax.core.spmd import logical_axis_rules as axis_rules
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.layers import attention_deepseek41
from maxtext.layers import moe
from maxtext.trainers.pre_train import train
from maxtext.trainers.pre_train import train_compile
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from tests.utils import deepseek_v41_frozen_topk

DEFAULT_CONFIG_PATH = "src/maxtext/configs/models/deepseek4.1-flash.yml"
BASE_CONFIG_PATH = "src/maxtext/configs/base.yml"
DEFAULT_MAX_HBM_GIB = 85.0
# Physical per-device HBM capacity enforced by XLA's TPU7x heap simulator (94.74 GiB).
TPU7X_HBM_CAPACITY_GIB = 94.74
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 94.74.
BENCH_MAX_HBM_GIB = TPU7X_HBM_CAPACITY_GIB
# TODO(yaoyuchen): Revisit 85.0 GiB minimal-to-full remat threshold once v7x compiler HBM scheduling improves.
REMAT_FALLBACK_HBM_GIB = 85.0
NAIVE_SUM_GATE_MODE = "naive_sum_budget"
OPTION_B_GATE_MODE = "option_b_tpu7x_heap_compile"
OPTION_B_BENCH_PRESETS: frozenset[str] = frozenset({"A8", "A8t", "A8b", "A8tb", "A9", "A9t", "A9b", "A9tb"})
PJRT_RUNTIME_GIB_BY_TOPOLOGY: dict[str, float] = {
    "tpu7x-128": 2.763,
    "tpu7x-256": 3.433,
}
LOAD_CEILING_GIB = 94.240
NEAR_CEILING_CAVEAT_WINDOW_GIB = 0.50
NEAR_CEILING_THRESHOLD_GIB = 93.740
MAX_NON_ALIASED_IO_GIB = 15.0
VERIFIED_TPU_LOAD_OBSERVATIONS: dict[tuple[str, str, str, bool], str] = {
    ("A8b", "tpu7x-128", "full", True): "ds41mm-w2-r4-nap",
}
VISION_MODULE_MARKERS: tuple[str, ...] = ("vision_encoder",)
VISION_COMPUTE_MARKERS: tuple[str, ...] = ("embed_patches", "merge_mm_embeddings")
VISION_HLO_MARKERS: tuple[str, ...] = VISION_MODULE_MARKERS + VISION_COMPUTE_MARKERS
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def _resolve_repo_head_commit_sha(
    repo_root: str | os.PathLike[str] | None = None,
    *,
    commit_sha_override: str | None = None,
) -> str:
  """Resolves the 40-char lowercase hex git HEAD commit SHA for `repo_root`, failing closed on error."""
  if commit_sha_override is not None:
    cleaned = str(commit_sha_override).strip().lower()
    if not _HEX40_RE.match(cleaned):
      raise ValueError(f"commit_sha_override must be a 40-character lowercase hex SHA, got {commit_sha_override!r}")
    return cleaned
  target_dir = os.fspath(repo_root) if repo_root is not None else _REPO_ROOT
  proc = subprocess.run(
      ["git", "-c", "safe.directory=*", "-C", target_dir, "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    raise RuntimeError(f"git rev-parse HEAD failed in {target_dir!r} (rc={proc.returncode}): {proc.stderr.strip()}")
  sha = proc.stdout.strip().lower()
  if not _HEX40_RE.match(sha):
    raise RuntimeError(f"git rev-parse HEAD in {target_dir!r} returned non-40-hex output: {proc.stdout!r}")
  return sha


class AotHbmGateError(AssertionError):
  """Raised when an AOT preset or benchmark batch fails its HBM budget gate."""

  def __init__(self, message: str, *, aot_result: dict[str, Any] | None = None) -> None:
    super().__init__(message)
    self.aot_result = aot_result


class AotCompileOomError(RuntimeError):
  """Raised when XLA `lowered.compile()` fails with HBM `RESOURCE_EXHAUSTED` / out-of-memory."""

  def __init__(self, message: str, *, aot_result: dict[str, Any] | None = None) -> None:
    super().__init__(message)
    self.aot_result = aot_result


_COMMON_FORWARD_OVERRIDES: dict[str, Any] = {
    "model_name": "deepseek4.1-flash",
    "override_model_config": True,
    "compile_topology": "tpu7x-128",
    "compile_topology_num_slices": 1,
    "per_device_batch_size": 0.125,
    "ici_fsdp_parallelism": -1,
    "ici_expert_parallelism": 16,
    "sparse_matmul": False,
    "capacity_factor": -1,
    "dtype": "float32",
    "weight_dtype": "bfloat16",
    "dtype_mm": "float32",
    "matmul_precision": "highest",
    "remat_policy": "none",
    "ds41_activation_quant": False,
    "use_multimodal": True,
    "vision_encoder_block": "deepseek41",
    "ds41_vision_grid": [39, 39],
}

_COMMON_TRAIN_OVERRIDES: dict[str, Any] = {
    "model_name": "deepseek4.1-flash",
    "override_model_config": True,
    "compile_topology": "tpu7x-128",
    "compile_topology_num_slices": 1,
    "max_target_length": 512,
    "per_device_batch_size": 0.125,
    "scan_layers": True,
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "dtype_mm": "bfloat16",
    "matmul_precision": "default",
    "remat_policy": "full",
    "ds41_activation_quant": True,
    "ici_fsdp_parallelism": -1,
    "ici_expert_parallelism": 16,
    "sparse_matmul": False,
    "capacity_factor": -1,
    "opt_type": "adamw",
    "use_multimodal": True,
    "vision_encoder_block": "deepseek41",
    "max_num_images_per_example": 1,
    "ds41_vision_grid": [39, 39],
    "dataset_type": "synthetic",
    "indexer_sparse_training": True,
    "indexer_loss_scaling_factor": 0.0,
    "learning_rate": 1e-5,
    "warmup_steps_fraction": 0.0,
    "adam_weight_decay": 0.1,
    "gradient_clipping_threshold": 1.0,
    "load_balance_loss_weight": 0.0001,
    "routed_bias_update_rate": 0.001,
}

_COMMON_BENCH_OVERRIDES: dict[str, Any] = {
    **_COMMON_TRAIN_OVERRIDES,
    "max_target_length": 4096,
    "per_device_batch_size": 1.0,
    "grad_dtype": "bfloat16",
    "num_vocab_tiling": 8,
    "quantization_local_shard_count": 8,
    "sparse_matmul": True,
    "megablox": True,
    "use_tokamax_gmm": True,
    "use_gmm_v2": True,
    "allow_split_physical_axes": True,
    "ragged_buffer_factor": 2.0,
    "ds41_activation_quant": False,
    "ds41_vision_grid": [39, 39],
    "enable_checkpointing": True,
    "async_checkpointing": False,
    "save_checkpoint_on_start": False,
    "save_checkpoint_on_completion": False,
    "checkpoint_period": 100000,
    "log_period": 1,
    "steps": 30,
    "learning_rate_schedule_steps": 30,
}

PRESET_SPECS: dict[str, dict[str, Any]] = {
    "A1": {
        "mode": "forward",
        "with_images": False,
        "num_images": 4,
        "expect_vision_in_hlo": False,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 1024,
        },
    },
    "A2": {
        "mode": "forward",
        "with_images": False,
        "num_images": 4,
        "expect_vision_in_hlo": False,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": False,
            "max_target_length": 1024,
        },
    },
    "A3": {
        "mode": "forward",
        "routing_mode": "natural",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 1024,
        },
    },
    "A3_frozen": {
        "mode": "forward",
        "routing_mode": "frozen_topk",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 1024,
        },
    },
    "A3_double": {
        "mode": "forward",
        "routing_mode": "double_pass",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 1024,
        },
    },
    "A3_frozen_b128": {
        "mode": "forward",
        "routing_mode": "frozen_topk",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 1024,
            "per_device_batch_size": 1.0,
        },
    },
    "A3_double_b128": {
        "mode": "forward",
        "routing_mode": "double_pass",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 1024,
            "per_device_batch_size": 1.0,
        },
    },
    "A4": {
        "mode": "forward",
        "routing_mode": "natural",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": False,
            "max_target_length": 1024,
        },
    },
    "A4_frozen": {
        "mode": "forward",
        "routing_mode": "frozen_topk",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": False,
            "max_target_length": 1024,
        },
    },
    "A4_double": {
        "mode": "forward",
        "routing_mode": "double_pass",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": False,
            "max_target_length": 1024,
        },
    },
    "A4_frozen_b128": {
        "mode": "forward",
        "routing_mode": "frozen_topk",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": False,
            "max_target_length": 1024,
            "per_device_batch_size": 1.0,
        },
    },
    "A4_double_b128": {
        "mode": "forward",
        "routing_mode": "double_pass",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": False,
            "max_target_length": 1024,
            "per_device_batch_size": 1.0,
        },
    },
    "A5": {
        "mode": "forward",
        "routing_mode": "natural",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 2048,
        },
    },
    "A5_frozen": {
        "mode": "forward",
        "routing_mode": "frozen_topk",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 2048,
        },
    },
    "A5_double": {
        "mode": "forward",
        "routing_mode": "double_pass",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 2048,
        },
    },
    "A5_frozen_b128": {
        "mode": "forward",
        "routing_mode": "frozen_topk",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 2048,
            "per_device_batch_size": 1.0,
        },
    },
    "A5_double_b128": {
        "mode": "forward",
        "routing_mode": "double_pass",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 2048,
            "per_device_batch_size": 1.0,
        },
    },
    "A5t": {
        "mode": "forward",
        "with_images": False,
        "num_images": 4,
        "expect_vision_in_hlo": False,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_FORWARD_OVERRIDES,
            "scan_layers": True,
            "max_target_length": 2048,
        },
    },
    "A6": {
        "mode": "train",
        "with_images": True,
        "num_images": 1,
        "expect_vision_in_hlo": True,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_TRAIN_OVERRIDES,
            "freeze_vision_encoder_params": True,
            "log_moe_bias_norms": True,
            "steps": 20,
            "learning_rate_schedule_steps": 20,
        },
    },
    "A7": {
        "mode": "train",
        "with_images": False,
        "num_images": 0,
        "expect_vision_in_hlo": False,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_TRAIN_OVERRIDES,
            "use_multimodal": True,
            "freeze_vision_encoder_params": True,
            "max_target_length": 1024,
            "opt_type": "sgd",
            "per_device_batch_size": 1.0,
        },
    },
    "A7b": {
        "mode": "train",
        "with_images": False,
        "num_images": 0,
        "expect_vision_in_hlo": False,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "max_hbm_gib": DEFAULT_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_TRAIN_OVERRIDES,
            "use_multimodal": True,
            "freeze_vision_encoder_params": True,
            "max_target_length": 1024,
            "opt_type": "sgd",
            "per_device_batch_size": 0.125,
        },
    },
    "A8": {
        "mode": "train",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": OPTION_B_GATE_MODE,
        "max_hbm_gib": BENCH_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_BENCH_OVERRIDES,
            "compile_topology": "tpu7x-128",
            "ici_fsdp_parallelism": 8,
            "ici_expert_parallelism": 16,
            "optimizer_memory_host_offload": True,
            "remat_policy": "minimal",
            "use_multimodal": True,
            "freeze_vision_encoder_params": True,
            "max_num_images_per_example": 4,
        },
    },
    "A8t": {
        "mode": "train",
        "with_images": False,
        "num_images": 0,
        "expect_vision_in_hlo": False,
        "gate_mode": OPTION_B_GATE_MODE,
        "max_hbm_gib": BENCH_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_BENCH_OVERRIDES,
            "compile_topology": "tpu7x-128",
            "ici_fsdp_parallelism": 8,
            "ici_expert_parallelism": 16,
            "optimizer_memory_host_offload": True,
            "remat_policy": "minimal",
            "use_multimodal": False,
            "freeze_vision_encoder_params": False,
        },
    },
    "A8b": {
        "mode": "train",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": OPTION_B_GATE_MODE,
        "max_hbm_gib": BENCH_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_BENCH_OVERRIDES,
            "compile_topology": "tpu7x-128",
            "ici_fsdp_parallelism": 8,
            "ici_expert_parallelism": 16,
            "optimizer_memory_host_offload": True,
            "remat_policy": "full",
            "use_multimodal": True,
            "freeze_vision_encoder_params": True,
            "max_num_images_per_example": 4,
        },
    },
    "A8tb": {
        "mode": "train",
        "with_images": False,
        "num_images": 0,
        "expect_vision_in_hlo": False,
        "gate_mode": OPTION_B_GATE_MODE,
        "max_hbm_gib": BENCH_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_BENCH_OVERRIDES,
            "compile_topology": "tpu7x-128",
            "ici_fsdp_parallelism": 8,
            "ici_expert_parallelism": 16,
            "optimizer_memory_host_offload": True,
            "remat_policy": "full",
            "use_multimodal": False,
            "freeze_vision_encoder_params": False,
        },
    },
    "A9": {
        "mode": "train",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": OPTION_B_GATE_MODE,
        "max_hbm_gib": BENCH_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_BENCH_OVERRIDES,
            "compile_topology": "tpu7x-256",
            "ici_fsdp_parallelism": 16,
            "ici_expert_parallelism": 16,
            "optimizer_memory_host_offload": True,
            "remat_policy": "minimal",
            "use_multimodal": True,
            "freeze_vision_encoder_params": True,
            "max_num_images_per_example": 4,
        },
    },
    "A9t": {
        "mode": "train",
        "with_images": False,
        "num_images": 0,
        "expect_vision_in_hlo": False,
        "gate_mode": OPTION_B_GATE_MODE,
        "max_hbm_gib": BENCH_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_BENCH_OVERRIDES,
            "compile_topology": "tpu7x-256",
            "ici_fsdp_parallelism": 16,
            "ici_expert_parallelism": 16,
            "optimizer_memory_host_offload": True,
            "remat_policy": "minimal",
            "use_multimodal": False,
            "freeze_vision_encoder_params": False,
        },
    },
    "A9b": {
        "mode": "train",
        "with_images": True,
        "num_images": 4,
        "expect_vision_in_hlo": True,
        "gate_mode": OPTION_B_GATE_MODE,
        "max_hbm_gib": BENCH_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_BENCH_OVERRIDES,
            "compile_topology": "tpu7x-256",
            "ici_fsdp_parallelism": 16,
            "ici_expert_parallelism": 16,
            "optimizer_memory_host_offload": True,
            "remat_policy": "full",
            "use_multimodal": True,
            "freeze_vision_encoder_params": True,
            "max_num_images_per_example": 4,
        },
    },
    "A9tb": {
        "mode": "train",
        "with_images": False,
        "num_images": 0,
        "expect_vision_in_hlo": False,
        "gate_mode": OPTION_B_GATE_MODE,
        "max_hbm_gib": BENCH_MAX_HBM_GIB,
        "overrides": {
            **_COMMON_BENCH_OVERRIDES,
            "compile_topology": "tpu7x-256",
            "ici_fsdp_parallelism": 16,
            "ici_expert_parallelism": 16,
            "optimizer_memory_host_offload": True,
            "remat_policy": "full",
            "use_multimodal": False,
            "freeze_vision_encoder_params": False,
        },
    },
}

W1_PRESET_ORDER: tuple[str, ...] = (
    "A1",
    "A2",
    "A3",
    "A3_frozen",
    "A3_double",
    "A3_frozen_b128",
    "A3_double_b128",
    "A4",
    "A4_frozen",
    "A4_double",
    "A4_frozen_b128",
    "A4_double_b128",
    "A5",
    "A5_frozen",
    "A5_double",
    "A5_frozen_b128",
    "A5_double_b128",
    "A5t",
    "A6",
)
W2_BENCH_PRESET_ORDER: tuple[str, ...] = ("A8", "A8t", "A8b", "A8tb")
W3_BENCH_PRESET_ORDER: tuple[str, ...] = ("A9", "A9t", "A9b", "A9tb")
BATCH_PRESET_ORDERS: dict[str, tuple[str, ...]] = {
    "W1_ALL": W1_PRESET_ORDER,
    "W2_BENCH_ALL": W2_BENCH_PRESET_ORDER,
    "W3_BENCH_ALL": W3_BENCH_PRESET_ORDER,
}
_BENCH_BATCH_PAIR_SPECS: dict[str, tuple[str, str, str, str]] = {
    "W2_BENCH_ALL": ("A8", "A8t", "A8b", "A8tb"),
    "W3_BENCH_ALL": ("A9", "A9t", "A9b", "A9tb"),
}


def get_preset_spec(
    preset_name: str,
    *,
    max_target_length_override: int | None = None,
) -> dict[str, Any]:
  """Returns a deep copy of a named AOT preset specification."""
  if preset_name not in PRESET_SPECS:
    raise KeyError(f"Unknown AOT preset {preset_name!r}; valid presets: {sorted(PRESET_SPECS)}")
  spec = copy.deepcopy(PRESET_SPECS[preset_name])
  if max_target_length_override is not None:
    spec["overrides"]["max_target_length"] = int(max_target_length_override)
  return spec


def _is_hbm_oom_exception(exc: BaseException) -> bool:
  """Returns True only for genuine XLA HBM `RESOURCE_EXHAUSTED` / out-of-memory runtime compile failures."""
  if not isinstance(exc, (jax.errors.JaxRuntimeError, RuntimeError)):
    return False
  msg_lower = str(exc).lower()
  return ("resource_exhausted" in msg_lower) or ("out of memory" in msg_lower)


def _resolve_mesh(config: Any, *, use_cpu_mesh: bool) -> jax.sharding.Mesh:
  """Constructs either the CPU device mesh or the target TPU topology mesh."""
  if use_cpu_mesh or not config.compile_topology:
    return maxtext_utils.get_mesh_from_config(config, jax.devices())
  train_compile.validate_config(config)
  return train_compile.get_topology_mesh(config)


def _extract_memory_stats(
    compiled: Any,
    *,
    max_hbm_gib: float = DEFAULT_MAX_HBM_GIB,
    option_b_heap_gate: bool = False,
) -> dict[str, Any]:
  """Extracts unclamped HBM component sizes and evaluates either the W1 `naive_sum` gate or Option B heap gate."""
  mem = compiled.memory_analysis()
  if mem is None:
    raise RuntimeError("compiled.memory_analysis() returned None; cannot verify HBM budget.")
  temp_bytes = int(mem.temp_size_in_bytes)
  arg_bytes = int(mem.argument_size_in_bytes)
  out_bytes = int(mem.output_size_in_bytes)
  alias_bytes = int(mem.alias_size_in_bytes)
  generated_code_bytes = (
      int(mem.generated_code_size_in_bytes)
      if hasattr(mem, "generated_code_size_in_bytes") and mem.generated_code_size_in_bytes is not None
      else 0
  )
  xla_peak_bytes = (
      int(mem.peak_memory_in_bytes)
      if hasattr(mem, "peak_memory_in_bytes") and mem.peak_memory_in_bytes is not None
      else None
  )
  non_aliased_io_bytes = max(0, arg_bytes + out_bytes - alias_bytes)
  naive_sum_bytes = temp_bytes + non_aliased_io_bytes
  gib = float(1024**3)
  temp_hbm_gib = temp_bytes / gib
  non_aliased_io_hbm_gib = non_aliased_io_bytes / gib
  naive_sum_hbm_gib = naive_sum_bytes / gib
  generated_code_hbm_gib = generated_code_bytes / gib
  xla_peak_hbm_gib = (xla_peak_bytes / gib) if xla_peak_bytes is not None else None
  max_hbm = float(max_hbm_gib)

  if option_b_heap_gate:
    passed = bool((temp_hbm_gib <= max_hbm) and (non_aliased_io_hbm_gib <= max_hbm))
    tpu7x_limit = max_hbm
    clamped_cap = round(min(tpu7x_limit, naive_sum_hbm_gib), 6)
    return {
        "compile_succeeded": True,
        "compile_error": "",
        "passed": passed,
        "gate_mode": OPTION_B_GATE_MODE,
        "tpu7x_heap_limit_gib": tpu7x_limit,
        "clamped_heap_cap_gib": clamped_cap,
        "temp_size_in_bytes": temp_bytes,
        "argument_size_in_bytes": arg_bytes,
        "output_size_in_bytes": out_bytes,
        "alias_size_in_bytes": alias_bytes,
        "generated_code_size_in_bytes": generated_code_bytes,
        "non_aliased_io_size_in_bytes": non_aliased_io_bytes,
        "temp_hbm_gib": temp_hbm_gib,
        "non_aliased_io_hbm_gib": non_aliased_io_hbm_gib,
        "naive_sum_hbm_gib": naive_sum_hbm_gib,
        "generated_code_hbm_gib": generated_code_hbm_gib,
        "xla_peak_memory_in_bytes": xla_peak_bytes,
        "xla_peak_hbm_gib": xla_peak_hbm_gib,
    }

  passed = bool(naive_sum_hbm_gib <= max_hbm)
  clamped_cap = round(min(TPU7X_HBM_CAPACITY_GIB, naive_sum_hbm_gib), 6)
  return {
      "compile_succeeded": True,
      "compile_error": "",
      "passed": passed,
      "gate_mode": NAIVE_SUM_GATE_MODE,
      "tpu7x_heap_limit_gib": TPU7X_HBM_CAPACITY_GIB,
      "clamped_heap_cap_gib": clamped_cap,
      "temp_size_in_bytes": temp_bytes,
      "argument_size_in_bytes": arg_bytes,
      "output_size_in_bytes": out_bytes,
      "alias_size_in_bytes": alias_bytes,
      "generated_code_size_in_bytes": generated_code_bytes,
      "non_aliased_io_size_in_bytes": non_aliased_io_bytes,
      "temp_hbm_gib": temp_hbm_gib,
      "non_aliased_io_hbm_gib": non_aliased_io_hbm_gib,
      "naive_sum_hbm_gib": naive_sum_hbm_gib,
      "generated_code_hbm_gib": generated_code_hbm_gib,
      "xla_peak_memory_in_bytes": xla_peak_bytes,
      "xla_peak_hbm_gib": xla_peak_hbm_gib,
      "peak_hbm_bytes": naive_sum_bytes,
      "peak_hbm_gib": naive_sum_hbm_gib,
  }


def _format_aot_log_line(preset_label: str, stats: Mapping[str, Any]) -> str:
  """Formats the standardized `[aot_check]` stdout status line with unclamped HBM components."""
  comp_ok = bool(stats["compile_succeeded"])
  passed = bool(stats["passed"])
  temp_val = stats["temp_hbm_gib"]
  io_val = stats["non_aliased_io_hbm_gib"]
  naive_val = stats["naive_sum_hbm_gib"]
  cap_val = stats["clamped_heap_cap_gib"]
  limit_val = float(stats["tpu7x_heap_limit_gib"])
  temp_s = "None" if temp_val is None else f"{float(temp_val):.6f}"
  io_s = "None" if io_val is None else f"{float(io_val):.6f}"
  naive_s = "None" if naive_val is None else f"{float(naive_val):.6f}"
  cap_s = "None" if cap_val is None else f"{float(cap_val):.6f}"
  line = (
      f"[aot_check] preset={preset_label} compile_succeeded={comp_ok} passed={passed} "
      f"temp_hbm_gib={temp_s} non_aliased_io_hbm_gib={io_s} "
      f"naive_sum_hbm_gib={naive_s} clamped_heap_cap_gib={cap_s} "
      f"tpu7x_heap_limit_gib={limit_val:.6f}"
  )
  if "load_sum_hbm_gib" in stats and stats["load_sum_hbm_gib"] is not None:
    caveat_val = stats["calibration_caveat"] if "calibration_caveat" in stats else None
    line += (
        f" load_sum_hbm_gib={float(stats['load_sum_hbm_gib']):.6f}"
        f" load_headroom_gib={float(stats['load_headroom_gib']):.6f}"
        f" opt_hbm_gib={float(stats['opt_hbm_gib']):.6f}"
        f" caveat={caveat_val}"
    )
  if "compile_error" in stats and stats["compile_error"]:
    line += f" error={stats['compile_error']}"
  return line


def _extract_optimizer_subtree(container: Any) -> Any:
  """Extracts the `.optimizer` pytree from a TrainState/nnx.State or Mapping, or returns None if absent."""
  if hasattr(container, "optimizer"):
    return container.optimizer
  if isinstance(container, Mapping) and "optimizer" in container:
    return container["optimizer"]
  return None


def _compute_optimizer_placement_stats(
    abstract_state: Any,
    state_mesh_shardings: Any,
) -> tuple[float, float]:
  """Computes per-device `(opt_hbm_gib, opt_pinned_host_gib)` from optimizer abstract state and shardings."""
  opt_shapes = _extract_optimizer_subtree(abstract_state)
  opt_shardings = _extract_optimizer_subtree(state_mesh_shardings)
  if opt_shapes is None or opt_shardings is None:
    raise ValueError("abstract_state or state_mesh_shardings is missing .optimizer pytree")

  shape_leaves = jax.tree_util.tree_leaves(opt_shapes)
  sharding_leaves = jax.tree_util.tree_leaves(opt_shardings)
  if len(shape_leaves) != len(sharding_leaves):
    raise ValueError(f"Optimizer shape/sharding leaf count mismatch: {len(shape_leaves)} != {len(sharding_leaves)}")
  if not shape_leaves:
    raise ValueError("Optimizer pytree has 0 leaves; expected positive optimizer state bytes")

  hbm_bytes = 0
  pinned_host_bytes = 0
  for idx, (leaf, sharding_spec) in enumerate(zip(shape_leaves, sharding_leaves)):
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype") or leaf.shape is None or leaf.dtype is None:
      raise ValueError(f"Optimizer leaf {idx} ({leaf!r}) is missing .shape or .dtype")
    shard_shape = sharding_spec.shard_shape(leaf.shape) if hasattr(sharding_spec, "shard_shape") else leaf.shape
    num_elems = 1
    for dim in shard_shape:
      num_elems *= int(dim)
    leaf_bytes = num_elems * int(jnp.dtype(leaf.dtype).itemsize)
    if hasattr(sharding_spec, "memory_kind") and sharding_spec.memory_kind is not None:
      mem_kind = str(sharding_spec.memory_kind).lower()
    elif hasattr(sharding_spec, "memory_space") and sharding_spec.memory_space is not None:
      mem_kind = str(sharding_spec.memory_space).lower()
    else:
      mem_kind = ""
    if "pinned_host" in mem_kind or "host" in mem_kind:
      pinned_host_bytes += leaf_bytes
    else:
      hbm_bytes += leaf_bytes

  if (hbm_bytes + pinned_host_bytes) <= 0:
    raise ValueError(
        f"Optimizer pytree total per-device bytes must be > 0, got hbm_bytes={hbm_bytes}, "
        f"pinned_host_bytes={pinned_host_bytes}"
    )
  gib = float(1024**3)
  return round(hbm_bytes / gib, 6), round(pinned_host_bytes / gib, 6)


def evaluate_load_time_hbm_gate(
    *,
    preset_name: str,
    topology: str,
    remat_policy: str,
    optimizer_memory_host_offload: bool,
    compile_succeeded: bool,
    temp_hbm_gib: float,
    non_aliased_io_hbm_gib: float,
    opt_hbm_gib: float,
    opt_pinned_host_gib: float = 0.0,
    generated_code_hbm_gib: float | None = None,
    max_hbm_gib: float = TPU7X_HBM_CAPACITY_GIB,
    verified_tpu_load_observations: Mapping[tuple[str, str, str, bool], str] | None = None,
) -> dict[str, Any]:
  """Evaluates the unclamped back-tested TPU7x `Executable::Load` HBM gate for W2/W3 presets."""
  if topology not in PJRT_RUNTIME_GIB_BY_TOPOLOGY:
    raise ValueError(
        f"Unsupported topology={topology!r} for load-time HBM gate; "
        f"expected one of {sorted(PJRT_RUNTIME_GIB_BY_TOPOLOGY)}"
    )
  pjrt_runtime_gib = float(PJRT_RUNTIME_GIB_BY_TOPOLOGY[topology])
  resident_live_gib = float(non_aliased_io_hbm_gib) + pjrt_runtime_gib
  # Unclamped bottom-of-HBM executable reservation estimate (never clamped at 74.88 GiB).
  if generated_code_hbm_gib is not None and float(generated_code_hbm_gib) > 0.0:
    bottom_reservation_est_gib = 0.8535 * (float(temp_hbm_gib) + float(generated_code_hbm_gib))
  else:
    bottom_reservation_est_gib = 0.861 * float(temp_hbm_gib)
  load_sum_hbm_gib = resident_live_gib + bottom_reservation_est_gib
  load_headroom_gib = LOAD_CEILING_GIB - load_sum_hbm_gib

  obs_map = VERIFIED_TPU_LOAD_OBSERVATIONS if verified_tpu_load_observations is None else verified_tpu_load_observations
  obs_key = (str(preset_name), str(topology), str(remat_policy), bool(optimizer_memory_host_offload))
  verified_obs = obs_map[obs_key] if obs_key in obs_map else None

  failure_reasons: list[str] = []
  if not compile_succeeded:
    failure_reasons.append("compile_failed")
  if float(temp_hbm_gib) > float(max_hbm_gib):
    failure_reasons.append(f"temp_hbm_gib ({float(temp_hbm_gib):.3f}) > max_hbm_gib ({float(max_hbm_gib):.3f})")
  if float(opt_hbm_gib) != 0.0:
    failure_reasons.append(f"opt_hbm_gib ({float(opt_hbm_gib):.6f}) != 0.0 (optimizer must be host-offloaded)")
  if bool(optimizer_memory_host_offload) and float(opt_pinned_host_gib) <= 0.0:
    failure_reasons.append(
        f"opt_pinned_host_gib ({float(opt_pinned_host_gib):.6f}) <= 0.0 "
        "(optimizer_memory_host_offload=True requires positive pinned_host optimizer state)"
    )
  if float(non_aliased_io_hbm_gib) > MAX_NON_ALIASED_IO_GIB:
    failure_reasons.append(f"non_aliased_io_hbm_gib ({float(non_aliased_io_hbm_gib):.3f}) > {MAX_NON_ALIASED_IO_GIB:.3f}")
  if load_sum_hbm_gib > LOAD_CEILING_GIB:
    failure_reasons.append(f"load_sum_hbm_gib ({load_sum_hbm_gib:.3f}) > load_ceiling_gib ({LOAD_CEILING_GIB:.3f})")

  calibration_caveat: str | None = None
  if NEAR_CEILING_THRESHOLD_GIB < load_sum_hbm_gib <= LOAD_CEILING_GIB:
    if verified_obs is not None:
      calibration_caveat = f"near_ceiling_verified_on_tpu:{verified_obs}"
    else:
      calibration_caveat = "near_ceiling_unverified_within_0.50_gib"
      failure_reasons.append(
          f"load_sum_hbm_gib ({load_sum_hbm_gib:.3f}) is within {NEAR_CEILING_CAVEAT_WINDOW_GIB:.2f} GiB "
          f"of {LOAD_CEILING_GIB:.3f} GiB ceiling without a live TPU load verification record for {obs_key}"
      )

  gate_passed = len(failure_reasons) == 0
  return {
      "gate_mode": "option_b_unclamped_load_time_hbm",
      "gate_passed": gate_passed,
      "failure_reasons": failure_reasons,
      "calibration_caveat": calibration_caveat,
      "verified_tpu_load_observation": verified_obs,
      "pjrt_runtime_gib": round(pjrt_runtime_gib, 6),
      "resident_live_gib": round(resident_live_gib, 6),
      "bottom_reservation_est_gib": round(bottom_reservation_est_gib, 6),
      "load_sum_hbm_gib": round(load_sum_hbm_gib, 6),
      "load_ceiling_gib": LOAD_CEILING_GIB,
      "near_ceiling_threshold_gib": NEAR_CEILING_THRESHOLD_GIB,
      "load_headroom_gib": round(load_headroom_gib, 6),
      "opt_hbm_gib": round(float(opt_hbm_gib), 6),
      "opt_pinned_host_gib": round(float(opt_pinned_host_gib), 6),
      "generated_code_hbm_gib": (None if generated_code_hbm_gib is None else round(float(generated_code_hbm_gib), 6)),
  }


def _compile_lowered_or_raise(
    lowered: Any,
    *,
    compiler_options: Mapping[str, Any],
    preset_label: str,
    max_hbm_gib: float,
    option_b_heap_gate: bool,
    base_failure_fields: Mapping[str, Any],
) -> Any:
  """Compiles `lowered`, raising `AotCompileOomError` on HBM OOM and propagating non-HBM errors untouched."""
  try:
    return lowered.compile(compiler_options=compiler_options)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    if not _is_hbm_oom_exception(exc):
      raise
    err_text = f"{type(exc).__name__}: {exc}"
    gate_mode = OPTION_B_GATE_MODE if option_b_heap_gate else NAIVE_SUM_GATE_MODE
    limit_gib = float(max_hbm_gib) if option_b_heap_gate else TPU7X_HBM_CAPACITY_GIB
    fail_result: dict[str, Any] = {
        **dict(base_failure_fields),
        "compile_succeeded": False,
        "compile_error": err_text,
        "passed": False,
        "gate_mode": gate_mode,
        "tpu7x_heap_limit_gib": limit_gib,
        "clamped_heap_cap_gib": None,
        "temp_hbm_gib": None,
        "non_aliased_io_hbm_gib": None,
        "naive_sum_hbm_gib": None,
        "max_hbm_gib": float(max_hbm_gib),
    }
    print(_format_aot_log_line(preset_label, fail_result), flush=True)
    raise AotCompileOomError(err_text, aot_result=fail_result) from exc


def _check_vision_hlo(
    compiled_hlo_text: str,
    *,
    expect_vision_in_hlo: bool,
    use_multimodal: bool = False,
    lowered_hlo_text: str = "",
) -> dict[str, bool]:
  """Checks vision module pytree presence and post-DCE compute markers in HLO text."""
  marker_presence: dict[str, bool] = {}
  for marker in VISION_MODULE_MARKERS:
    marker_presence[marker] = (marker in compiled_hlo_text) or (bool(lowered_hlo_text) and (marker in lowered_hlo_text))
  for marker in VISION_COMPUTE_MARKERS:
    marker_presence[marker] = marker in compiled_hlo_text
  if expect_vision_in_hlo and not all(marker_presence.values()):
    missing = [m for m in VISION_HLO_MARKERS if not marker_presence[m]]
    raise AssertionError(
        f"Expected all vision HLO markers {VISION_HLO_MARKERS} in compiled HLO when "
        f"expect_vision_in_hlo=True, missing={missing}, got presence={marker_presence}."
    )
  if not expect_vision_in_hlo:
    compute_present = [m for m in VISION_COMPUTE_MARKERS if marker_presence[m]]
    if compute_present:
      raise AssertionError(
          f"Expected no vision compute markers {VISION_COMPUTE_MARKERS} in compiled HLO when "
          f"expect_vision_in_hlo=False, found compute_present={compute_present}, got presence={marker_presence}."
      )
    if not use_multimodal and marker_presence["vision_encoder"]:
      raise AssertionError(
          f"Expected no vision_encoder marker in compiled HLO when use_multimodal=False, "
          f"got presence={marker_presence}."
      )
  return marker_presence


VALID_FORWARD_ROUTING_MODES: tuple[str, ...] = ("natural", "frozen_topk", "natural_trace", "double_pass")
_AOT_FROZEN_TOPK_ATTR = "_frozen_topk_indices"
_AOT_FROZEN_INDEXER_TOPK_ATTR = "_frozen_indexer_topk"
_AOT_CAPTURED_TOPK_ATTR = "_captured_topk_indices"
_AOT_CAPTURED_INDEXER_TOPK_ATTR = "_captured_indexer_topk"


def _resolve_abstract_replicated_sharding(model: Any) -> Any:
  """Returns `NamedSharding(mesh, PartitionSpec())` when `model` has `jax.ShapeDtypeStruct` state leaves."""
  for leaf in jax.tree_util.tree_leaves(nnx.state(model)):
    if isinstance(leaf, jax.ShapeDtypeStruct):
      sharding_obj = getattr(leaf, "sharding", None)
      if isinstance(sharding_obj, jax.sharding.NamedSharding):
        return jax.sharding.NamedSharding(sharding_obj.mesh, jax.sharding.PartitionSpec())
      return sharding_obj
  return None


@contextlib.contextmanager
def _abstract_frozen_topk_routing_context(
    abs_model: Any,
    config: Any,
    *,
    batch_size: int,
    seq_len: int,
) -> Iterator[None]:
  """Attaches abstract `ShapeDtypeStruct` teacher-forced MoE/Indexer routing buffers for `frozen_topk` AOT lowering."""
  replicated_sharding = _resolve_abstract_replicated_sharding(abs_model)
  top_k = int(config.num_experts_per_tok)
  idx_k = int(config.indexer_topk)
  (
      unscanned_targets,
      scanned_targets,
      _,
      unscanned_idx_targets,
      scanned_idx_targets,
      _,
  ) = deepseek_v41_frozen_topk._collect_decoder_moe_and_indexer_targets(  # pylint: disable=protected-access
      abs_model.decoder, config
  )
  patched_blocks: list[moe.RoutedMoE] = []
  patched_indexers: list[attention_deepseek41.DeepSeek41Indexer] = []
  orig_deepseek_routing = moe.RoutedMoE.deepseek_routing
  orig_indexer_call = attention_deepseek41.DeepSeek41Indexer.__call__

  def _frozen_deepseek_routing(
      self_moe: moe.RoutedMoE,
      gate_logits: jax.Array,
      pre_bias_logits: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    if not hasattr(self_moe, _AOT_FROZEN_TOPK_ATTR) or getattr(self_moe, _AOT_FROZEN_TOPK_ATTR) is None:
      return orig_deepseek_routing(self_moe, gate_logits, pre_bias_logits)
    with jax.named_scope("ds41_frozen_topk_moe"):
      frozen_var = getattr(self_moe, _AOT_FROZEN_TOPK_ATTR)
      top_k_indices = jnp.asarray(frozen_var[...], dtype=jnp.int32)
      expected_idx_shape = pre_bias_logits.shape[:-1] + (self_moe.num_experts_per_tok,)
      if top_k_indices.shape != expected_idx_shape:
        raise ValueError(f"FrozenTopK indices shape {top_k_indices.shape} != expected {expected_idx_shape}")
      if top_k_indices.ndim == 3:
        top_k_indices = self_moe._maybe_shard_with_logical(  # pylint: disable=protected-access
            top_k_indices,
            ("activation_batch_moe", "activation_length_moe", None),
        )
      tag_routing_fn = moe.routing_tag_fn(self_moe.config)
      if tag_routing_fn is not None:
        top_k_indices = tag_routing_fn(top_k_indices)
      use_dense_vjp = bool(self_moe.config.router_topk_matmul_vjp)
      if use_dense_vjp:
        top_k_weights = moe.take_along_last_axis_dense_vjp(pre_bias_logits, top_k_indices, pre_bias_logits.shape[-1])
      else:
        top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
      return top_k_weights, top_k_indices

  def _frozen_indexer_call(  # pylint: disable=too-many-positional-arguments
      self_idx: attention_deepseek41.DeepSeek41Indexer,
      x: jax.Array,
      qr: jax.Array,
      positions: jax.Array,
      shared: dict[str, Any],
      decoder_segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, Any]]:
    if not hasattr(self_idx, _AOT_FROZEN_INDEXER_TOPK_ATTR) or getattr(self_idx, _AOT_FROZEN_INDEXER_TOPK_ATTR) is None:
      return orig_indexer_call(self_idx, x, qr, positions, shared, decoder_segment_ids=decoder_segment_ids)
    with jax.named_scope("ds41_frozen_topk_indexer"):
      _, reachable, compress_lens, num_compressed, updates = (
          deepseek_v41_frozen_topk._compute_indexer_scores_and_reachability(  # pylint: disable=protected-access
              self_idx, x, qr, positions, shared, decoder_segment_ids=decoder_segment_ids
          )
      )
      frozen_idx_var = getattr(self_idx, _AOT_FROZEN_INDEXER_TOPK_ATTR)
      frozen_top_idx = jnp.asarray(frozen_idx_var[...], dtype=jnp.int32)
      k = min(int(self_idx.index_topk), num_compressed)
      top_idx = frozen_top_idx[..., :k]
      valid_slot = (top_idx >= 0) & (top_idx < num_compressed)
      safe_top_idx = jnp.where(valid_slot, top_idx, 0)
      if decoder_segment_ids is None:
        keep = valid_slot & (safe_top_idx < compress_lens[..., None])
      else:
        keep = valid_slot & jnp.take_along_axis(reachable, safe_top_idx, axis=-1)
      return attention_deepseek41._scatter_bool(safe_top_idx, keep, num_compressed), updates  # pylint: disable=protected-access

  try:
    if bool(config.scan_layers):
      for segment, _, moe_block in scanned_targets:
        moe_aval = jax.ShapeDtypeStruct(
            (int(segment.repeats), int(batch_size), int(seq_len), top_k),
            jnp.int32,
            sharding=replicated_sharding,
        )
        setattr(moe_block, _AOT_FROZEN_TOPK_ATTR, nnx.Variable(moe_aval))
        patched_blocks.append(moe_block)
      for segment, _, idx_mod in scanned_idx_targets:
        idx_aval = jax.ShapeDtypeStruct(
            (int(segment.repeats), int(batch_size), int(seq_len), idx_k),
            jnp.int32,
            sharding=replicated_sharding,
        )
        setattr(idx_mod, _AOT_FROZEN_INDEXER_TOPK_ATTR, nnx.Variable(idx_aval))
        patched_indexers.append(idx_mod)
    else:
      for _, moe_block in unscanned_targets:
        moe_aval = jax.ShapeDtypeStruct(
            (int(batch_size), int(seq_len), top_k),
            jnp.int32,
            sharding=replicated_sharding,
        )
        setattr(moe_block, _AOT_FROZEN_TOPK_ATTR, nnx.Variable(moe_aval))
        patched_blocks.append(moe_block)
      for _, idx_mod in unscanned_idx_targets:
        idx_aval = jax.ShapeDtypeStruct(
            (int(batch_size), int(seq_len), idx_k),
            jnp.int32,
            sharding=replicated_sharding,
        )
        setattr(idx_mod, _AOT_FROZEN_INDEXER_TOPK_ATTR, nnx.Variable(idx_aval))
        patched_indexers.append(idx_mod)
    moe.RoutedMoE.deepseek_routing = _frozen_deepseek_routing  # type: ignore[method-assign]
    attention_deepseek41.DeepSeek41Indexer.__call__ = _frozen_indexer_call  # type: ignore[method-assign]
    yield
  finally:
    moe.RoutedMoE.deepseek_routing = orig_deepseek_routing  # type: ignore[method-assign]
    attention_deepseek41.DeepSeek41Indexer.__call__ = orig_indexer_call  # type: ignore[method-assign]
    for moe_block in patched_blocks:
      if hasattr(moe_block, _AOT_FROZEN_TOPK_ATTR):
        delattr(moe_block, _AOT_FROZEN_TOPK_ATTR)
    for idx_mod in patched_indexers:
      if hasattr(idx_mod, _AOT_FROZEN_INDEXER_TOPK_ATTR):
        delattr(idx_mod, _AOT_FROZEN_INDEXER_TOPK_ATTR)


@contextlib.contextmanager
def _abstract_natural_routing_trace_context(
    abs_model: Any,
    config: Any,
    *,
    batch_size: int,
    seq_len: int,
) -> Iterator[None]:
  """Attaches abstract `ShapeDtypeStruct` natural-routing trace buffers for `natural_trace` AOT lowering."""
  replicated_sharding = _resolve_abstract_replicated_sharding(abs_model)
  top_k = int(config.num_experts_per_tok)
  index_topk = int(config.indexer_topk)
  (
      unscanned_targets,
      scanned_targets,
      _,
      unscanned_idx_targets,
      scanned_idx_targets,
      _,
  ) = deepseek_v41_frozen_topk._collect_decoder_moe_and_indexer_targets(  # pylint: disable=protected-access
      abs_model.decoder, config
  )
  patched_blocks: list[moe.RoutedMoE] = []
  patched_indexers: list[attention_deepseek41.DeepSeek41Indexer] = []
  orig_deepseek_routing = moe.RoutedMoE.deepseek_routing
  orig_indexer_call = attention_deepseek41.DeepSeek41Indexer.__call__

  def _trace_deepseek_routing(
      self_moe: moe.RoutedMoE,
      gate_logits: jax.Array,
      pre_bias_logits: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    with jax.named_scope("ds41_natural_trace_moe"):
      top_k_weights, top_k_indices = orig_deepseek_routing(self_moe, gate_logits, pre_bias_logits)
      if hasattr(self_moe, _AOT_CAPTURED_TOPK_ATTR) and getattr(self_moe, _AOT_CAPTURED_TOPK_ATTR) is not None:
        getattr(self_moe, _AOT_CAPTURED_TOPK_ATTR).set_value(jnp.asarray(top_k_indices, dtype=jnp.int32))
      return top_k_weights, top_k_indices

  def _trace_indexer_call(  # pylint: disable=too-many-positional-arguments
      self_idx: attention_deepseek41.DeepSeek41Indexer,
      x: jax.Array,
      qr: jax.Array,
      positions: jax.Array,
      shared: dict[str, Any],
      decoder_segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, Any]]:
    if (
        not hasattr(self_idx, _AOT_CAPTURED_INDEXER_TOPK_ATTR)
        or getattr(self_idx, _AOT_CAPTURED_INDEXER_TOPK_ATTR) is None
    ):
      return orig_indexer_call(self_idx, x, qr, positions, shared, decoder_segment_ids=decoder_segment_ids)
    with jax.named_scope("ds41_natural_trace_indexer"):
      scores, reachable, compress_lens, num_compressed, updates = (
          deepseek_v41_frozen_topk._compute_indexer_scores_and_reachability(  # pylint: disable=protected-access
              self_idx, x, qr, positions, shared, decoder_segment_ids=decoder_segment_ids
          )
      )
      k = min(int(self_idx.index_topk), num_compressed)
      top_idx = jax.lax.top_k(scores, k)[1]
      keep = (
          top_idx < compress_lens[..., None]
          if decoder_segment_ids is None
          else jnp.take_along_axis(reachable, top_idx, axis=-1)
      )
      valid_idx = jnp.where(keep, top_idx.astype(jnp.int32), jnp.int32(-1))
      if k < int(self_idx.index_topk):
        pad_width = ((0, 0), (0, 0), (0, int(self_idx.index_topk) - int(k)))
        valid_idx = jnp.pad(valid_idx, pad_width, constant_values=-1)
      else:
        valid_idx = valid_idx[..., : int(self_idx.index_topk)]
      getattr(self_idx, _AOT_CAPTURED_INDEXER_TOPK_ATTR).set_value(valid_idx)
      return attention_deepseek41._scatter_bool(top_idx, keep, num_compressed), updates  # pylint: disable=protected-access

  try:
    if bool(config.scan_layers):
      for segment, _, moe_block in scanned_targets:
        init_moe = jax.ShapeDtypeStruct(
            (int(segment.repeats), int(batch_size), int(seq_len), top_k),
            jnp.int32,
            sharding=replicated_sharding,
        )
        setattr(moe_block, _AOT_CAPTURED_TOPK_ATTR, nnx.Variable(init_moe))
        patched_blocks.append(moe_block)
      for segment, _, idx_mod in scanned_idx_targets:
        init_idx = jax.ShapeDtypeStruct(
            (int(segment.repeats), int(batch_size), int(seq_len), index_topk),
            jnp.int32,
            sharding=replicated_sharding,
        )
        setattr(idx_mod, _AOT_CAPTURED_INDEXER_TOPK_ATTR, nnx.Variable(init_idx))
        patched_indexers.append(idx_mod)
    else:
      for _, moe_block in unscanned_targets:
        init_moe = jax.ShapeDtypeStruct(
            (int(batch_size), int(seq_len), top_k),
            jnp.int32,
            sharding=replicated_sharding,
        )
        setattr(moe_block, _AOT_CAPTURED_TOPK_ATTR, nnx.Variable(init_moe))
        patched_blocks.append(moe_block)
      for _, idx_mod in unscanned_idx_targets:
        init_idx = jax.ShapeDtypeStruct(
            (int(batch_size), int(seq_len), index_topk),
            jnp.int32,
            sharding=replicated_sharding,
        )
        setattr(idx_mod, _AOT_CAPTURED_INDEXER_TOPK_ATTR, nnx.Variable(init_idx))
        patched_indexers.append(idx_mod)
    moe.RoutedMoE.deepseek_routing = _trace_deepseek_routing  # type: ignore[method-assign]
    attention_deepseek41.DeepSeek41Indexer.__call__ = _trace_indexer_call  # type: ignore[method-assign]
    yield
  finally:
    moe.RoutedMoE.deepseek_routing = orig_deepseek_routing  # type: ignore[method-assign]
    attention_deepseek41.DeepSeek41Indexer.__call__ = orig_indexer_call  # type: ignore[method-assign]
    for moe_block in patched_blocks:
      if hasattr(moe_block, _AOT_CAPTURED_TOPK_ATTR):
        delattr(moe_block, _AOT_CAPTURED_TOPK_ATTR)
    for idx_mod in patched_indexers:
      if hasattr(idx_mod, _AOT_CAPTURED_INDEXER_TOPK_ATTR):
        delattr(idx_mod, _AOT_CAPTURED_INDEXER_TOPK_ATTR)


def run_forward_aot_check(
    config: Any,
    *,
    with_images: bool,
    num_images: int = 4,
    expect_vision_in_hlo: bool | None = None,
    max_hbm_gib: float = DEFAULT_MAX_HBM_GIB,
    option_b_heap_gate: bool = False,
    preset_label: str = "forward",
    use_cpu_mesh: bool = False,
    routing_mode: str = "natural",
) -> dict[str, Any]:
  """Lowers and compiles a pure forward pass, verifying full [B, S, V] logits, vision HLO, routing mode, and HBM."""
  if routing_mode not in VALID_FORWARD_ROUTING_MODES:
    raise ValueError(f"Unsupported routing_mode {routing_mode!r}; expected one of {VALID_FORWARD_ROUTING_MODES}")

  if routing_mode == "double_pass":
    frozen_res: dict[str, Any]
    natural_trace_res: dict[str, Any]
    try:
      frozen_res = run_forward_aot_check(
          config,
          with_images=with_images,
          num_images=num_images,
          expect_vision_in_hlo=expect_vision_in_hlo,
          max_hbm_gib=max_hbm_gib,
          option_b_heap_gate=option_b_heap_gate,
          preset_label=f"{preset_label}:frozen_topk",
          use_cpu_mesh=use_cpu_mesh,
          routing_mode="frozen_topk",
      )
    except AotHbmGateError as exc:
      if exc.aot_result is None:
        raise
      frozen_res = dict(exc.aot_result)
    try:
      natural_trace_res = run_forward_aot_check(
          config,
          with_images=with_images,
          num_images=num_images,
          expect_vision_in_hlo=expect_vision_in_hlo,
          max_hbm_gib=max_hbm_gib,
          option_b_heap_gate=option_b_heap_gate,
          preset_label=f"{preset_label}:natural_trace",
          use_cpu_mesh=use_cpu_mesh,
          routing_mode="natural_trace",
      )
    except AotHbmGateError as exc:
      if exc.aot_result is None:
        raise
      natural_trace_res = dict(exc.aot_result)

    frozen_peak_hbm_gib = float(frozen_res["peak_hbm_gib"])
    natural_trace_peak_hbm_gib = float(natural_trace_res["peak_hbm_gib"])
    frozen_temp_gib = float(frozen_res["temp_hbm_gib"])
    natural_trace_temp_gib = float(natural_trace_res["temp_hbm_gib"])
    frozen_io_gib = float(frozen_res["non_aliased_io_hbm_gib"])
    natural_trace_io_gib = float(natural_trace_res["non_aliased_io_hbm_gib"])
    frozen_gen_code_gib = float(frozen_res.get("generated_code_hbm_gib", 0.0) or 0.0)
    natural_trace_gen_code_gib = float(natural_trace_res.get("generated_code_hbm_gib", 0.0) or 0.0)

    combined_temp_hbm_gib = max(frozen_temp_gib, natural_trace_temp_gib)
    combined_non_aliased_io_hbm_gib = max(frozen_io_gib, natural_trace_io_gib) + abs(natural_trace_io_gib - frozen_io_gib)
    shared_weights_double_pass_peak_hbm_gib = (
        combined_temp_hbm_gib + combined_non_aliased_io_hbm_gib + frozen_gen_code_gib + natural_trace_gen_code_gib
    )
    unshared_sum_double_pass_hbm_gib = frozen_peak_hbm_gib + natural_trace_peak_hbm_gib
    max_hbm = float(max_hbm_gib)
    passed = bool(
        (frozen_peak_hbm_gib <= max_hbm)
        and (natural_trace_peak_hbm_gib <= max_hbm)
        and (shared_weights_double_pass_peak_hbm_gib <= max_hbm)
    )
    clamped_cap = round(min(TPU7X_HBM_CAPACITY_GIB, shared_weights_double_pass_peak_hbm_gib), 6)
    gib = float(1024**3)
    commit_sha = (
        str(frozen_res["git_commit_sha"])
        if "git_commit_sha" in frozen_res and frozen_res["git_commit_sha"]
        else _resolve_repo_head_commit_sha()
    )
    double_result: dict[str, Any] = {
        "preset": preset_label,
        "git_commit_sha": commit_sha,
        "mode": "forward",
        "routing_mode": "double_pass",
        "with_images": bool(with_images),
        "num_images": int(num_images) if with_images else 0,
        "scan_layers": bool(config.scan_layers),
        "batch_size": int(frozen_res["batch_size"]),
        "seq_len": int(frozen_res["seq_len"]),
        "vocab_size": int(frozen_res["vocab_size"]),
        "moe_layer_count": int(frozen_res["moe_layer_count"]),
        "indexer_layer_count": int(frozen_res["indexer_layer_count"]),
        "output_shape": list(frozen_res["output_shape"]),
        "output_dtype": str(frozen_res["output_dtype"]),
        "vision_hlo_markers": dict(frozen_res["vision_hlo_markers"]),
        "max_hbm_gib": max_hbm,
        "compile_succeeded": bool(frozen_res["compile_succeeded"] and natural_trace_res["compile_succeeded"]),
        "compile_error": "",
        "passed": passed,
        "gate_mode": NAIVE_SUM_GATE_MODE,
        "tpu7x_heap_limit_gib": TPU7X_HBM_CAPACITY_GIB,
        "clamped_heap_cap_gib": clamped_cap,
        "temp_hbm_gib": combined_temp_hbm_gib,
        "non_aliased_io_hbm_gib": combined_non_aliased_io_hbm_gib,
        "naive_sum_hbm_gib": shared_weights_double_pass_peak_hbm_gib,
        "generated_code_hbm_gib": frozen_gen_code_gib + natural_trace_gen_code_gib,
        "peak_hbm_bytes": int(round(shared_weights_double_pass_peak_hbm_gib * gib)),
        "peak_hbm_gib": shared_weights_double_pass_peak_hbm_gib,
        "frozen_peak_hbm_gib": frozen_peak_hbm_gib,
        "natural_trace_peak_hbm_gib": natural_trace_peak_hbm_gib,
        "shared_weights_double_pass_peak_hbm_gib": shared_weights_double_pass_peak_hbm_gib,
        "unshared_sum_double_pass_hbm_gib": unshared_sum_double_pass_hbm_gib,
        "frozen_pass": frozen_res,
        "natural_trace_pass": natural_trace_res,
    }
    print(_format_aot_log_line(preset_label, double_result), flush=True)
    if not passed:
      raise AotHbmGateError(
          f"Double-pass forward AOT HBM exceeded budget {max_hbm:.3f} GiB on preset={preset_label}: "
          f"frozen_peak_hbm_gib={frozen_peak_hbm_gib:.6f}, "
          f"natural_trace_peak_hbm_gib={natural_trace_peak_hbm_gib:.6f}, "
          f"shared_weights_double_pass_peak_hbm_gib={shared_weights_double_pass_peak_hbm_gib:.6f}.",
          aot_result=double_result,
      )
    return double_result

  jax.clear_caches()
  expected_vision = bool(with_images) if expect_vision_in_hlo is None else bool(expect_vision_in_hlo)
  mesh = _resolve_mesh(config, use_cpu_mesh=use_cpu_mesh)
  _, abs_model = model_creation_utils.create_nnx_abstract_model(config, mesh=mesh)
  batch_size = int(config.global_batch_size_to_load)
  seq_len = int(config.max_target_length)
  vocab_size = int(config.vocab_size)

  (
      _,
      _,
      moe_layer_ids,
      _,
      _,
      indexer_layer_ids,
  ) = deepseek_v41_frozen_topk._collect_decoder_moe_and_indexer_targets(  # pylint: disable=protected-access
      abs_model.decoder, config
  )

  routing_ctx: Any = contextlib.nullcontext()
  if routing_mode == "frozen_topk":
    routing_ctx = _abstract_frozen_topk_routing_context(
        abs_model,
        config,
        batch_size=batch_size,
        seq_len=seq_len,
    )
  elif routing_mode == "natural_trace":
    routing_ctx = _abstract_natural_routing_trace_context(
        abs_model,
        config,
        batch_size=batch_size,
        seq_len=seq_len,
    )

  with routing_ctx:
    graphdef, abs_state = nnx.split(abs_model)
    state_sharding = jax.tree.map(lambda x: x.sharding, abs_state)
    data_sharding = sharding.get_input_data_sharding(config, mesh)

    tokens = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32, sharding=data_sharding)
    positions = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32, sharding=data_sharding)
    segment_ids = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32, sharding=data_sharding)

    if with_images:
      n_vit_h, n_vit_w = config.ds41_vision_grid
      patch_dim = int(config.num_channels_for_vit) * (int(config.patch_size_for_vit) ** 2)
      images = jax.ShapeDtypeStruct(
          (batch_size, int(num_images), int(n_vit_h) * int(n_vit_w), patch_dim),
          jnp.float32,
          sharding=data_sharding,
      )
      image_masks = jax.ShapeDtypeStruct(
          (batch_size, int(num_images)),
          jnp.int32,
          sharding=data_sharding,
      )
      in_shardings = (state_sharding, data_sharding, data_sharding, data_sharding, data_sharding, data_sharding)
    else:
      images = None
      image_masks = None
      in_shardings = (state_sharding, data_sharding, data_sharding, data_sharding, None, None)

    def _forward_fn(state, tok, pos, seg, imgs, img_masks):  # pylint: disable=too-many-positional-arguments
      model = nnx.merge(graphdef, state)
      logits = model(
          decoder_input_tokens=tok,
          decoder_positions=pos,
          enable_dropout=False,
          decoder_segment_ids=seg,
          encoder_images=imgs,
          encoder_image_masks=img_masks,
      )
      if routing_mode == "natural_trace":
        _, _, captured_non_params = nnx.split(model, nnx.Param, ...)
        return logits, captured_non_params
      return logits

    with jax.set_mesh(mesh), axis_rules(config.logical_axis_rules):
      jitted = jax.jit(_forward_fn, in_shardings=in_shardings, out_shardings=None)
      out_aval = jax.eval_shape(_forward_fn, abs_state, tokens, positions, segment_ids, images, image_masks)
      lowered = jitted.lower(abs_state, tokens, positions, segment_ids, images, image_masks)

  logits_aval = out_aval[0] if routing_mode == "natural_trace" else out_aval
  expected_out_shape = (batch_size, seq_len, vocab_size)
  if tuple(logits_aval.shape) != expected_out_shape:
    raise AssertionError(
        f"Forward AOT output shape must be full [B, S, V] = {expected_out_shape} "
        f"(never last-token sliced [B, V]), got {tuple(logits_aval.shape)}."
    )

  compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
  lowered_hlo_text = lowered.as_text()

  base_meta: dict[str, Any] = {
      "preset": preset_label,
      "git_commit_sha": _resolve_repo_head_commit_sha(),
      "mode": "forward",
      "routing_mode": routing_mode,
      "with_images": bool(with_images),
      "num_images": int(num_images) if with_images else 0,
      "scan_layers": bool(config.scan_layers),
      "batch_size": batch_size,
      "seq_len": seq_len,
      "vocab_size": vocab_size,
      "moe_layer_count": len(moe_layer_ids),
      "indexer_layer_count": len(indexer_layer_ids),
  }
  compiled = _compile_lowered_or_raise(
      lowered,
      compiler_options=compiler_options,
      preset_label=preset_label,
      max_hbm_gib=float(max_hbm_gib),
      option_b_heap_gate=option_b_heap_gate,
      base_failure_fields=base_meta,
  )
  mem_stats = _extract_memory_stats(
      compiled,
      max_hbm_gib=float(max_hbm_gib),
      option_b_heap_gate=option_b_heap_gate,
  )
  print(_format_aot_log_line(preset_label, mem_stats), flush=True)
  compiled_hlo_text = compiled.as_text()
  combined_hlo_text = f"{lowered_hlo_text}\n{compiled_hlo_text}"
  if routing_mode == "frozen_topk":
    if "ds41_frozen_topk_moe" not in combined_hlo_text or "ds41_frozen_topk_indexer" not in combined_hlo_text:
      raise AssertionError(
          f"Expected ds41_frozen_topk_moe and ds41_frozen_topk_indexer in compiled HLO for preset={preset_label} "
          "(routing_mode='frozen_topk')."
      )
  elif routing_mode == "natural_trace":
    if "ds41_natural_trace_moe" not in combined_hlo_text or "ds41_natural_trace_indexer" not in combined_hlo_text:
      raise AssertionError(
          f"Expected ds41_natural_trace_moe and ds41_natural_trace_indexer in compiled HLO for preset={preset_label} "
          "(routing_mode='natural_trace')."
      )
  marker_presence = _check_vision_hlo(
      compiled_hlo_text,
      expect_vision_in_hlo=expected_vision,
      use_multimodal=bool(config.use_multimodal),
      lowered_hlo_text=lowered_hlo_text,
  )
  result: dict[str, Any] = {
      **base_meta,
      "output_shape": list(logits_aval.shape),
      "output_dtype": str(logits_aval.dtype),
      "vision_hlo_markers": marker_presence,
      "max_hbm_gib": float(max_hbm_gib),
      **mem_stats,
  }
  if routing_mode == "frozen_topk":
    result["frozen_topk_hlo_verified"] = True
  elif routing_mode == "natural_trace":
    result["natural_trace_hlo_verified"] = True
  if not mem_stats["passed"]:
    raise AotHbmGateError(
        f"Forward AOT HBM exceeded budget {float(max_hbm_gib):.3f} GiB on preset={preset_label}: "
        f"temp_hbm_gib={mem_stats['temp_hbm_gib']:.6f}, "
        f"non_aliased_io_hbm_gib={mem_stats['non_aliased_io_hbm_gib']:.6f}, "
        f"naive_sum_hbm_gib={mem_stats['naive_sum_hbm_gib']:.6f}.",
        aot_result=result,
    )
  return result


def run_train_aot_check(
    config: Any,
    *,
    with_images: bool = True,
    expect_vision_in_hlo: bool | None = None,
    max_hbm_gib: float = DEFAULT_MAX_HBM_GIB,
    remat_fallback_hbm_gib: float = REMAT_FALLBACK_HBM_GIB,
    option_b_heap_gate: bool = False,
    preset_label: str = "train",
    use_cpu_mesh: bool = False,
) -> dict[str, Any]:
  """Lowers and compiles `train.train_step`, verifying vision HLO and HBM budget."""
  jax.clear_caches()
  expected_vision = bool(with_images) if expect_vision_in_hlo is None else bool(expect_vision_in_hlo)
  mesh = _resolve_mesh(config, use_cpu_mesh=use_cpu_mesh)
  (
      shaped_train_args,
      shaped_train_kwargs,
      state_mesh_shardings,
      _,
      model,
  ) = train_compile.get_shaped_inputs(mesh, config)
  data_sharding = sharding.get_input_data_sharding(config, mesh)
  shaped_batch = dict(shaped_train_args[1])

  if with_images and config.use_multimodal:
    if config.dataset_type == "synthetic":
      if "image_masks" in shaped_batch:
        del shaped_batch["image_masks"]
    else:
      batch_size = int(config.global_batch_size_to_load)
      num_images = int(config.max_num_images_per_example)
      n_vit_h, n_vit_w = config.ds41_vision_grid
      patch_dim = int(config.num_channels_for_vit) * (int(config.patch_size_for_vit) ** 2)
      shaped_batch["images"] = jax.ShapeDtypeStruct(
          (batch_size, num_images, int(n_vit_h) * int(n_vit_w), patch_dim),
          jnp.float32,
          sharding=data_sharding,
      )
      shaped_batch["image_masks"] = jax.ShapeDtypeStruct(
          (batch_size, num_images),
          jnp.int32,
          sharding=data_sharding,
      )
  else:
    if "images" in shaped_batch:
      del shaped_batch["images"]
    if "image_masks" in shaped_batch:
      del shaped_batch["image_masks"]

  shaped_train_args = (shaped_train_args[0], shaped_batch)
  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
  input_state_mesh_shardings = sharding.build_zero1_input_state_mesh_shardings(
      config, state_mesh_shardings, params_shardings
  )
  (
      func_to_compile,
      in_shard,
      out_shard,
      static_argnums,
      donate_argnums,
  ) = maxtext_utils.get_functional_train_with_signature(
      train.train_step,
      data_sharding,
      input_state_mesh_shardings,
      model,
      config,
      params_shardings,
  )
  prev_step0_env = os.environ["DS41_LOG_STEP0_MOE_DROPPED"] if "DS41_LOG_STEP0_MOE_DROPPED" in os.environ else None
  if option_b_heap_gate and prev_step0_env is None:
    os.environ["DS41_LOG_STEP0_MOE_DROPPED"] = "1"
  try:
    mesh_ctx = mesh if config.enable_diloco else contextlib.nullcontext()
    with jax.set_mesh(mesh), mesh_ctx, axis_rules(config.logical_axis_rules):
      jitted = jax.jit(
          func_to_compile,
          in_shardings=in_shard,
          out_shardings=out_shard,
          static_argnums=static_argnums,
          donate_argnums=donate_argnums,
      )
      maxtext_utils.maybe_dump_jaxpr(config, jitted, shaped_train_args)
      lowered = jitted.lower(*shaped_train_args, **shaped_train_kwargs)
  finally:
    if option_b_heap_gate and prev_step0_env is None and "DS41_LOG_STEP0_MOE_DROPPED" in os.environ:
      del os.environ["DS41_LOG_STEP0_MOE_DROPPED"]

  opt_hbm_gib, opt_pinned_host_gib = _compute_optimizer_placement_stats(
      shaped_train_args[0],
      input_state_mesh_shardings,
  )
  lowered_hlo_text = lowered.as_text()
  compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
  base_meta: dict[str, Any] = {
      "preset": preset_label,
      "git_commit_sha": _resolve_repo_head_commit_sha(),
      "mode": "train",
      "with_images": bool(with_images),
      "use_multimodal": bool(config.use_multimodal),
      "freeze_vision_encoder_params": bool(config.freeze_vision_encoder_params),
      "scan_layers": bool(config.scan_layers),
      "compile_topology": str(config.compile_topology),
      "remat_policy": str(config.remat_policy),
      "sparse_matmul": bool(config.sparse_matmul),
      "optimizer_memory_host_offload": bool(config.optimizer_memory_host_offload),
      "opt_hbm_gib": opt_hbm_gib,
      "opt_pinned_host_gib": opt_pinned_host_gib,
      "max_target_length": int(config.max_target_length),
      "per_device_batch_size": float(config.per_device_batch_size),
      "remat_fallback_hbm_gib": float(remat_fallback_hbm_gib),
  }
  compiled = _compile_lowered_or_raise(
      lowered,
      compiler_options=compiler_options,
      preset_label=preset_label,
      max_hbm_gib=float(max_hbm_gib),
      option_b_heap_gate=option_b_heap_gate,
      base_failure_fields=base_meta,
  )
  mem_stats = _extract_memory_stats(
      compiled,
      max_hbm_gib=float(max_hbm_gib),
      option_b_heap_gate=option_b_heap_gate,
  )
  compiled_hlo_text = compiled.as_text()
  marker_presence = _check_vision_hlo(
      compiled_hlo_text,
      expect_vision_in_hlo=expected_vision,
      use_multimodal=bool(config.use_multimodal),
      lowered_hlo_text=lowered_hlo_text,
  )
  remat_fb = float(remat_fallback_hbm_gib)
  recommended_remat = (
      "minimal"
      if (
          bool(mem_stats["passed"])
          and float(mem_stats["temp_hbm_gib"]) <= remat_fb
          and float(mem_stats["naive_sum_hbm_gib"]) <= remat_fb
      )
      else "full"
  )
  load_gate_fields: dict[str, Any] = {}
  topo_str = str(config.compile_topology)
  if option_b_heap_gate and not use_cpu_mesh and topo_str in PJRT_RUNTIME_GIB_BY_TOPOLOGY:
    load_eval = evaluate_load_time_hbm_gate(
        preset_name=preset_label,
        topology=topo_str,
        remat_policy=str(config.remat_policy),
        optimizer_memory_host_offload=bool(config.optimizer_memory_host_offload),
        compile_succeeded=bool(mem_stats["compile_succeeded"]),
        temp_hbm_gib=float(mem_stats["temp_hbm_gib"]),
        non_aliased_io_hbm_gib=float(mem_stats["non_aliased_io_hbm_gib"]),
        opt_hbm_gib=opt_hbm_gib,
        opt_pinned_host_gib=opt_pinned_host_gib,
        generated_code_hbm_gib=(mem_stats["generated_code_hbm_gib"] if "generated_code_hbm_gib" in mem_stats else None),
        max_hbm_gib=float(max_hbm_gib),
    )
    load_gate_fields.update(
        {
            "pjrt_runtime_gib": load_eval["pjrt_runtime_gib"],
            "resident_live_gib": load_eval["resident_live_gib"],
            "bottom_reservation_est_gib": load_eval["bottom_reservation_est_gib"],
            "load_sum_hbm_gib": load_eval["load_sum_hbm_gib"],
            "load_ceiling_gib": load_eval["load_ceiling_gib"],
            "near_ceiling_threshold_gib": load_eval["near_ceiling_threshold_gib"],
            "load_headroom_gib": load_eval["load_headroom_gib"],
            "calibration_caveat": load_eval["calibration_caveat"],
            "verified_tpu_load_observation": load_eval["verified_tpu_load_observation"],
            "load_gate_passed": load_eval["gate_passed"],
            "load_gate_failure_reasons": load_eval["failure_reasons"],
        }
    )
    mem_stats["passed"] = bool(mem_stats["passed"] and load_eval["gate_passed"])

  result: dict[str, Any] = {
      **base_meta,
      "recommended_remat_policy": recommended_remat,
      "vision_hlo_markers": marker_presence,
      "max_hbm_gib": float(max_hbm_gib),
      **mem_stats,
      **load_gate_fields,
  }
  print(_format_aot_log_line(preset_label, result), flush=True)
  if not mem_stats["passed"]:
    raise AotHbmGateError(
        f"Train AOT HBM gate failed on preset={preset_label}: "
        f"temp_hbm_gib={mem_stats['temp_hbm_gib']:.6f}, "
        f"non_aliased_io_hbm_gib={mem_stats['non_aliased_io_hbm_gib']:.6f}, "
        f"naive_sum_hbm_gib={mem_stats['naive_sum_hbm_gib']:.6f}, "
        f"load_sum_hbm_gib={result['load_sum_hbm_gib'] if 'load_sum_hbm_gib' in result else None}, "
        f"reasons={result['load_gate_failure_reasons'] if 'load_gate_failure_reasons' in result else None}.",
        aot_result=result,
    )
  return result


def _build_config_from_overrides(
    overrides: Mapping[str, Any],
    *,
    cpu_mini: bool = False,
) -> Any:
  """Initializes a `MaxTextConfig` from either the 749B YAML or the 14-layer CPU mini-twin."""
  if cpu_mini:
    from tests.utils import deepseek_v41_mm_testing as mm_testing  # pylint: disable=import-outside-toplevel

    mini_kw: dict[str, Any] = {
        "vocab_size": 256,
        "ds41_image_token_id": 200,
        "use_multimodal": bool(overrides["use_multimodal"]) if "use_multimodal" in overrides else True,
        "scan_layers": bool(overrides["scan_layers"]) if "scan_layers" in overrides else True,
        "dtype": str(overrides["dtype"]) if "dtype" in overrides else "float32",
        "weight_dtype": str(overrides["weight_dtype"]) if "weight_dtype" in overrides else "bfloat16",
        "dtype_mm": str(overrides["dtype_mm"]) if "dtype_mm" in overrides else "float32",
        "matmul_precision": str(overrides["matmul_precision"]) if "matmul_precision" in overrides else "highest",
        "remat_policy": str(overrides["remat_policy"]) if "remat_policy" in overrides else "none",
        "ds41_activation_quant": (
            bool(overrides["ds41_activation_quant"]) if "ds41_activation_quant" in overrides else False
        ),
        "per_device_batch_size": (
            max(1.0, float(overrides["per_device_batch_size"])) if "per_device_batch_size" in overrides else 1.0
        ),
        "max_target_length": int(overrides["max_target_length"]) if "max_target_length" in overrides else 64,
        "freeze_vision_encoder_params": (
            bool(overrides["freeze_vision_encoder_params"]) if "freeze_vision_encoder_params" in overrides else False
        ),
        "max_num_images_per_example": (
            int(overrides["max_num_images_per_example"]) if "max_num_images_per_example" in overrides else 1
        ),
        "opt_type": str(overrides["opt_type"]) if "opt_type" in overrides else "adamw",
        "optimizer_memory_host_offload": (
            bool(overrides["optimizer_memory_host_offload"]) if "optimizer_memory_host_offload" in overrides else False
        ),
        "log_moe_bias_norms": (bool(overrides["log_moe_bias_norms"]) if "log_moe_bias_norms" in overrides else False),
        "dataset_type": str(overrides["dataset_type"]) if "dataset_type" in overrides else "tfds",
    }
    for extra_key, cast_fn in (
        ("learning_rate", float),
        ("warmup_steps_fraction", float),
        ("adam_weight_decay", float),
        ("gradient_clipping_threshold", float),
        ("load_balance_loss_weight", float),
        ("routed_bias_update_rate", float),
        ("steps", int),
        ("learning_rate_schedule_steps", int),
        ("grad_dtype", str),
        ("num_vocab_tiling", int),
        ("quantization_local_shard_count", int),
        ("sparse_matmul", bool),
        ("allow_split_physical_axes", bool),
        ("ragged_buffer_factor", float),
        ("ds41_activation_quant", bool),
        ("optimizer_memory_host_offload", bool),
        ("enable_checkpointing", bool),
        ("async_checkpointing", bool),
        ("save_checkpoint_on_start", bool),
        ("save_checkpoint_on_completion", bool),
        ("checkpoint_period", int),
        ("log_period", int),
    ):
      if extra_key in overrides:
        mini_kw[extra_key] = cast_fn(overrides[extra_key])
    if bool(mini_kw["sparse_matmul"]) if "sparse_matmul" in mini_kw else False:
      # XLA:CPU ThunkEmitter lacks `ragged-all-to-all` and TPU Mosaic GMM kernels; use local `jax.lax.ragged_dot` on CPU.
      mini_kw["megablox"] = False
      mini_kw["use_tokamax_gmm"] = False
      mini_kw["use_gmm_v2"] = False
      mini_kw["ragged_buffer_factor"] = -1.0
    if "ds41_attention_chunk" in overrides:
      raw_chunk = int(overrides["ds41_attention_chunk"])
      mini_kw["ds41_attention_chunk"] = min(raw_chunk, int(mini_kw["max_target_length"])) if raw_chunk > 0 else 0
    return mm_testing.mini_vision_maxtext_config(**mini_kw)
  return pyconfig.initialize([sys.argv[0], BASE_CONFIG_PATH], **dict(overrides))


def _run_bench_batch_preset(
    preset_name: str,
    *,
    cpu_mini: bool = False,
    max_target_length_override: int | None = None,
    max_hbm_gib_override: float | None = None,
    remat_fallback_hbm_gib_override: float | None = None,
) -> dict[str, Any]:
  """Evaluates minimal + full remat benchmark preset pairs sequentially, failing closed on non-HBM or full-remat errors."""
  if preset_name in ("all", "ALL"):
    preset_seq: tuple[str, ...] = ("A8", "A8t", "A8b", "A8tb", "A9", "A9t", "A9b", "A9tb")
  else:
    preset_seq = _BENCH_BATCH_PAIR_SPECS[preset_name]
  commit_sha = _resolve_repo_head_commit_sha()
  remat_fb_hbm = (
      float(REMAT_FALLBACK_HBM_GIB) if remat_fallback_hbm_gib_override is None else float(remat_fallback_hbm_gib_override)
  )
  results: dict[str, dict[str, Any]] = {}
  for p in preset_seq:
    spec_p = get_preset_spec(p, max_target_length_override=max_target_length_override)
    max_hbm_p = float(spec_p["max_hbm_gib"]) if max_hbm_gib_override is None else float(max_hbm_gib_override)
    try:
      res_p = dict(
          run_preset(
              p,
              cpu_mini=cpu_mini,
              max_target_length_override=max_target_length_override,
              max_hbm_gib_override=max_hbm_gib_override,
              remat_fallback_hbm_gib_override=remat_fallback_hbm_gib_override,
          )
      )
      res_p.setdefault("git_commit_sha", commit_sha)
      results[p] = res_p
    except (AotHbmGateError, AotCompileOomError) as exc:
      if exc.aot_result is None:
        raise
      err_res = {**dict(exc.aot_result), "preset": p}
      err_res.setdefault("git_commit_sha", commit_sha)
      results[p] = err_res
    except (jax.errors.JaxRuntimeError, RuntimeError) as exc:
      if not _is_hbm_oom_exception(exc):
        raise
      err_text = f"{type(exc).__name__}: {exc}"
      fail_entry: dict[str, Any] = {
          "preset": p,
          "git_commit_sha": commit_sha,
          "mode": str(spec_p["mode"]),
          "with_images": bool(spec_p["with_images"]),
          "use_multimodal": bool(spec_p["overrides"]["use_multimodal"]),
          "compile_topology": str(spec_p["overrides"]["compile_topology"]),
          "remat_policy": str(spec_p["overrides"]["remat_policy"]),
          "max_hbm_gib": max_hbm_p,
          "remat_fallback_hbm_gib": remat_fb_hbm,
          "gate_mode": OPTION_B_GATE_MODE,
          "tpu7x_heap_limit_gib": max_hbm_p,
          "clamped_heap_cap_gib": None,
          "temp_hbm_gib": None,
          "non_aliased_io_hbm_gib": None,
          "naive_sum_hbm_gib": None,
          "compile_succeeded": False,
          "passed": False,
          "compile_error": err_text,
      }
      print(_format_aot_log_line(p, fail_entry), flush=True)
      results[p] = fail_entry

  if len(preset_seq) == 8:
    min_pair_ok = all(
        bool(results[p]["passed"])
        and results[p].get("temp_hbm_gib") is not None
        and float(results[p]["temp_hbm_gib"]) <= remat_fb_hbm
        and float(results[p]["naive_sum_hbm_gib"]) <= remat_fb_hbm
        for p in ("A8", "A8t", "A9", "A9t")
    )
    full_pair_ok = all(bool(results[p]["passed"]) for p in ("A8b", "A8tb", "A9b", "A9tb"))
  else:
    mm_min, txt_min, mm_full, txt_full = preset_seq
    min_pair_ok = bool(
        results[mm_min]["passed"]
        and results[txt_min]["passed"]
        and results[mm_min]["temp_hbm_gib"] is not None
        and results[txt_min]["temp_hbm_gib"] is not None
        and float(results[mm_min]["temp_hbm_gib"]) <= remat_fb_hbm
        and float(results[mm_min]["naive_sum_hbm_gib"]) <= remat_fb_hbm
        and float(results[txt_min]["temp_hbm_gib"]) <= remat_fb_hbm
        and float(results[txt_min]["naive_sum_hbm_gib"]) <= remat_fb_hbm
    )
    full_pair_ok = bool(results[mm_full]["passed"] and results[txt_full]["passed"])
  if min_pair_ok:
    selected_remat_policy: str | None = "minimal"
    batch_passed = True
  elif full_pair_ok:
    selected_remat_policy = "full"
    batch_passed = True
  else:
    selected_remat_policy = None
    batch_passed = False

  batch_result: dict[str, Any] = {
      "preset": preset_name,
      "git_commit_sha": commit_sha,
      "passed": batch_passed,
      "selected_remat_policy": selected_remat_policy,
      "remat_fallback_hbm_gib": remat_fb_hbm,
      "results": results,
  }
  if not batch_passed:
    raise AotHbmGateError(
        f"Benchmark batch {preset_name} failed Option B HBM gate.",
        aot_result=batch_result,
    )
  return batch_result


def run_preset(
    preset_name: str,
    *,
    cpu_mini: bool = False,
    max_target_length_override: int | None = None,
    max_hbm_gib_override: float | None = None,
    remat_fallback_hbm_gib_override: float | None = None,
    expect_vision_in_hlo_override: bool | None = None,
) -> dict[str, Any]:
  """Executes a single AOT preset (`A1..A9tb`) or a batch preset (`W1_ALL`, `W2_BENCH_ALL`, `W3_BENCH_ALL`, `all`)."""
  if preset_name in _BENCH_BATCH_PAIR_SPECS or preset_name in ("all", "ALL"):
    return _run_bench_batch_preset(
        preset_name,
        cpu_mini=cpu_mini,
        max_target_length_override=max_target_length_override,
        max_hbm_gib_override=max_hbm_gib_override,
        remat_fallback_hbm_gib_override=remat_fallback_hbm_gib_override,
    )
  if preset_name in BATCH_PRESET_ORDERS:
    commit_sha = _resolve_repo_head_commit_sha()
    results = {}
    for p in BATCH_PRESET_ORDERS[preset_name]:
      res_p = dict(
          run_preset(
              p,
              cpu_mini=cpu_mini,
              max_target_length_override=max_target_length_override,
              max_hbm_gib_override=max_hbm_gib_override,
              remat_fallback_hbm_gib_override=remat_fallback_hbm_gib_override,
          )
      )
      res_p.setdefault("git_commit_sha", commit_sha)
      results[p] = res_p
    return {"preset": preset_name, "git_commit_sha": commit_sha, "passed": True, "results": results}

  spec = get_preset_spec(preset_name, max_target_length_override=max_target_length_override)
  cfg = _build_config_from_overrides(spec["overrides"], cpu_mini=cpu_mini)
  max_hbm = float(spec["max_hbm_gib"]) if max_hbm_gib_override is None else float(max_hbm_gib_override)
  remat_fb_hbm = (
      float(REMAT_FALLBACK_HBM_GIB) if remat_fallback_hbm_gib_override is None else float(remat_fallback_hbm_gib_override)
  )
  expect_vision = (
      bool(spec["expect_vision_in_hlo"]) if expect_vision_in_hlo_override is None else bool(expect_vision_in_hlo_override)
  )
  option_b_gate = preset_name in OPTION_B_BENCH_PRESETS
  mode = spec["mode"]
  if mode in ("forward", "decode"):
    res = run_forward_aot_check(
        cfg,
        with_images=bool(spec["with_images"]),
        num_images=int(spec["num_images"]),
        expect_vision_in_hlo=expect_vision,
        max_hbm_gib=max_hbm,
        option_b_heap_gate=option_b_gate,
        preset_label=preset_name,
        use_cpu_mesh=cpu_mini,
        routing_mode=str(spec.get("routing_mode", "natural")),
    )
    res["preset"] = preset_name
    res["requested_mode"] = mode
    return res
  if mode == "train":
    res = run_train_aot_check(
        cfg,
        with_images=bool(spec["with_images"]),
        expect_vision_in_hlo=expect_vision,
        max_hbm_gib=max_hbm,
        remat_fallback_hbm_gib=remat_fb_hbm,
        option_b_heap_gate=option_b_gate,
        preset_label=preset_name,
        use_cpu_mesh=cpu_mini,
    )
    res["preset"] = preset_name
    return res
  raise ValueError(f"Unsupported AOT mode {mode!r} in preset {preset_name!r}")


def _str2bool(val: str) -> bool:
  lower = val.strip().lower()
  if lower in ("true", "1", "yes", "y"):
    return True
  if lower in ("false", "0", "no", "n"):
    return False
  raise argparse.ArgumentTypeError(f"Expected boolean string, got {val!r}")


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash forward/train/decode AOT HBM & HLO checker.")
  parser.add_argument(
      "--preset",
      type=str,
      default="A3",
      help="Preset name (A1..A9tb, W1_ALL, W2_BENCH_ALL, or W3_BENCH_ALL).",
  )
  parser.add_argument("--cpu_mini", type=_str2bool, default=False, help="Run against 14-layer CPU mini-twin.")
  parser.add_argument("--max_target_length", type=int, default=None, help="Optional max_target_length override.")
  parser.add_argument("--max_hbm_gib", type=float, default=None, help="Optional HBM ceiling in GiB.")
  parser.add_argument(
      "--remat_fallback_hbm_gib",
      type=float,
      default=None,
      help="HBM threshold in GiB above which recommended_remat_policy switches from minimal to full.",
  )
  parser.add_argument(
      "--expect_vision_in_hlo",
      type=_str2bool,
      default=None,
      help="Optional override for expected vision encoder presence in compiled HLO.",
  )
  parser.add_argument(
      "--output_json",
      "--output-json",
      dest="output_json",
      type=str,
      default=None,
      help="Optional path to write JSON report.",
  )
  args = parser.parse_args(argv)

  try:
    result = run_preset(
        args.preset,
        cpu_mini=args.cpu_mini,
        max_target_length_override=args.max_target_length,
        max_hbm_gib_override=args.max_hbm_gib,
        remat_fallback_hbm_gib_override=args.remat_fallback_hbm_gib,
        expect_vision_in_hlo_override=args.expect_vision_in_hlo,
    )
  except (AotHbmGateError, AotCompileOomError) as exc:
    if exc.aot_result is not None:
      rendered_err = json.dumps(exc.aot_result, indent=2, sort_keys=True)
      print(rendered_err, flush=True)
      if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
          f.write(rendered_err + "\n")
    raise

  rendered = json.dumps(result, indent=2, sort_keys=True)
  print(rendered, flush=True)
  if args.output_json:
    with open(args.output_json, "w", encoding="utf-8") as f:
      f.write(rendered + "\n")
  if "passed" in result and not bool(result["passed"]):
    raise AotHbmGateError(f"Preset {args.preset!r} reported passed=False.", aot_result=result)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
