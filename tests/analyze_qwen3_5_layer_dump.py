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

"""Standalone CLI tool for dumping and analyzing all intermediate tensors

from 1 decoder layer of Qwen3.5 MoE between MaxText Training and vLLM Inference.

Usage:
  python3 tests/analyze_qwen3_5_layer_dump.py --dtype=bfloat16 --output_dir=/tmp/qwen3_5_dumps
  python3 tests/analyze_qwen3_5_layer_dump.py --dtype=float32 --output_dir=/tmp/qwen3_5_dumps
"""

import argparse
import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import qwen3_5
from maxtext.utils import maxtext_utils
from tests.unit.qwen3_5_layer_dump_test import (capture_qwen3_5_layer_intermediates, compute_drift_metrics,
                                                dump_tensors_to_npz, generate_comparison_markdown_table,
                                                sync_qwen3_5_layer_weights)
from tests.utils.test_helpers import get_test_config_path


def parse_args():
    """Parses command line arguments for the Qwen3.5 layer dump tool."""
    parser = argparse.ArgumentParser(
        description="Qwen3.5 MoE 1-Layer Intermediate Tensor Dump Tool"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"]
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=2048)
    parser.add_argument("--moe_mlp_dim", type=int, default=512)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_experts_per_tok", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="/tmp/qwen3_5_layer_dumps")
    return parser.parse_args()


def main():
    """Executes 1-layer forward pass for both training and inference configurations and outputs drift table."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(
        "================================================================================"
    )
    print("QWEN3.5 MoE 1-LAYER INTERMEDIATE TENSOR DUMP & DRIFT ANALYSIS")
    print("  Training:  attention='flash'    | sparse_matmul=True")
    print("  Inference: attention='vllm_rpa' | fused_moe_matmul=True (vLLM)")
    print(f"  DType:     {args.dtype}")
    print(
        f"  Shape:     Batch={args.batch_size}, SeqLen={args.seq_len}, EmbDim={args.emb_dim}"
    )
    print(
        "================================================================================\n"
    )

    base_kwargs = {
        "override_model_config": True,
        "num_decoder_layers": 1,
        "model_name": "qwen3.5-35b-a3b",
        "base_emb_dim": args.emb_dim,
        "base_mlp_dim": args.moe_mlp_dim,
        "base_moe_mlp_dim": args.moe_mlp_dim,
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "vocab_size": 32000,
        "max_target_length": args.seq_len,
        "max_prefill_predict_length": args.seq_len,
        "per_device_batch_size": 1.0,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
        "scan_layers": False,
        "enable_checkpointing": False,
        "log_config": False,
        "inhomogeneous_layer_cycle_interval": 1,  # Layer 0 is full attention + MoE
    }

    print("Initializing Training Configuration...")
    cfg_train = pyconfig.initialize(
        [sys.argv[0], get_test_config_path(), "attention=flash", "sparse_matmul=True"],
        weight_dtype=args.dtype,
        dtype=args.dtype,
        **base_kwargs,
    )

    print("Initializing Inference Configuration...")
    cfg_infer = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path("inference/vllm.yml"),
            "attention=vllm_rpa",
            "prefuse_moe_weights=True",
            "model_call_mode=inference",
            "ici_data_parallelism=-1",
        ],
        weight_dtype=args.dtype,
        dtype=args.dtype,
        **base_kwargs,
    )

    train_devices = maxtext_utils.create_device_mesh(cfg_train)
    train_mesh = Mesh(train_devices, cfg_train.mesh_axes)

    infer_devices = maxtext_utils.create_device_mesh(cfg_infer)
    infer_mesh = Mesh(infer_devices, cfg_infer.mesh_axes)

    num_devices = len(jax.devices())
    actual_batch_size = max(num_devices, 4)

    print("Instantiating NNX Qwen3_5DecoderLayer instances...")
    rng = nnx.Rngs(params=42)
    train_layer = qwen3_5.Qwen3_5DecoderLayer(
        config=cfg_train,
        mesh=train_mesh,
        model_mode=MODEL_MODE_TRAIN,
        layer_idx=0,
        rngs=rng,
    )
    infer_layer = qwen3_5.Qwen3_5DecoderLayer(
        config=cfg_infer,
        mesh=infer_mesh,
        model_mode=MODEL_MODE_PREFILL,
        layer_idx=0,
        rngs=rng,
    )

    print("Synchronizing identical parameter matrices from Trainer to Inference...")
    sync_qwen3_5_layer_weights(train_layer, infer_layer)

    # Prepare synthetic input
    dtype_jax = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    key = jax.random.PRNGKey(101)
    inputs = jax.random.normal(
        key, (actual_batch_size, args.seq_len, args.emb_dim), dtype=dtype_jax
    )
    decoder_positions = jnp.broadcast_to(
        jnp.arange(args.seq_len, dtype=jnp.int32), (actual_batch_size, args.seq_len)
    )
    decoder_segment_ids = jnp.ones((actual_batch_size, args.seq_len), dtype=jnp.int32)

    inputs = jax.device_put(
        inputs, NamedSharding(train_mesh, P(("data", "fsdp"), None, None))
    )
    decoder_positions = jax.device_put(
        decoder_positions, NamedSharding(train_mesh, P(("data", "fsdp"), None))
    )
    decoder_segment_ids = jax.device_put(
        decoder_segment_ids, NamedSharding(train_mesh, P(("data", "fsdp"), None))
    )

    print("Executing Training forward pass & capturing all sub-tensors...")
    _, train_tensors = capture_qwen3_5_layer_intermediates(
        train_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_TRAIN,
    )

    print("Executing Inference forward pass & capturing all sub-tensors...")
    _, infer_tensors = capture_qwen3_5_layer_intermediates(
        infer_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_PREFILL,
    )

    print(f"Captured {len(train_tensors)} intermediate tensors from Training.")
    print(f"Captured {len(infer_tensors)} intermediate tensors from Inference.")

    metrics = {}
    for name, t_train in train_tensors.items():
        metrics[name] = compute_drift_metrics(t_train, infer_tensors[name])

    table_md = generate_comparison_markdown_table(metrics)
    print("\n" + table_md + "\n")

    # Dump archives
    train_dump_file = os.path.join(
        args.output_dir, f"qwen3_5_layer_train_{args.dtype}.npz"
    )
    infer_dump_file = os.path.join(
        args.output_dir, f"qwen3_5_layer_infer_{args.dtype}.npz"
    )

    print(f"Saving training tensors to:  {train_dump_file}")
    dump_tensors_to_npz(train_tensors, train_dump_file)

    print(f"Saving inference tensors to: {infer_dump_file}")
    dump_tensors_to_npz(infer_tensors, infer_dump_file)

    print("\n[SUCCESS] Intermediate tensor dump and comparison completed successfully.")


if __name__ == "__main__":
    main()
