# Copyright 2026 Google LLC
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

"""Diagnostic script to isolate intrinsic error vs cascaded amplification in T19 and T20."""

import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import jax
from jax import numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.models import qwen3_5
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
from tests.unit.qwen3_5_layer_dump_test import (
    sync_qwen3_5_layer_weights,
    capture_qwen3_5_layer_intermediates,
)


def compute_metrics(a, b):
    a_np = np.array(jax.device_get(a), dtype=np.float32)
    b_np = np.array(jax.device_get(b), dtype=np.float32)
    abs_diff = np.abs(a_np - b_np)
    return {
        "max_abs_err": float(np.max(abs_diff)),
        "mae": float(np.mean(abs_diff)),
        "cos_sim": float(np.dot(a_np.ravel(), b_np.ravel()) / (np.linalg.norm(a_np.ravel()) * np.linalg.norm(b_np.ravel()) + 1e-12)),
    }


def run_isolation_diagnostics():
    batch_size = 4
    seq_len = 512
    emb_dim = 2048
    moe_mlp_dim = 512
    num_experts = 8
    num_experts_per_tok = 8
    dtype_str = "float32"

    base_kwargs = {
        "override_model_config": True,
        "model_name": "qwen3.5-35b-a3b",
        "base_emb_dim": emb_dim,
        "base_num_query_heads": 16,
        "base_num_kv_heads": 2,
        "head_dim": 256,
        "base_mlp_dim": moe_mlp_dim,
        "moe_mlp_dim": moe_mlp_dim,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "max_target_length": seq_len,
        "max_prefill_predict_length": seq_len,
        "per_device_batch_size": 1.0,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
        "scan_layers": False,
        "enable_checkpointing": False,
        "log_config": False,
        "weight_dtype": dtype_str,
        "dtype": dtype_str,
        "inhomogeneous_layer_cycle_interval": 1,
    }

    cfg_train = pyconfig.initialize(
        [sys.argv[0], get_test_config_path(), "attention=flash", "use_tokamax_splash=True", "sa_use_base2_exp=False", "sparse_matmul=False"],
        **base_kwargs,
    )

    # Tensor-parallel degree for the inference mesh is capped at
    # `num_kv_heads` (2 for qwen3.5-35b-a3b): the RPA kernel's shard_map
    # requires the (batch_size+1,)-shaped `query_start_loc` and (3,)-shaped
    # `request_distribution` AttentionMetadata arrays to be evenly divisible
    # by the mesh axis size, which a 4-device data-parallel mesh violates
    # for batch_size=4. See tests/run_qwen3_5_logit_parity.py for the
    # reference pattern.
    infer_tp_degree = min(len(jax.devices()), cfg_train.num_kv_heads)

    cfg_infer = pyconfig.initialize(
        [sys.argv[0], get_test_config_path("inference/vllm.yml"), "attention=vllm_rpa", "prefuse_moe_weights=False", "model_call_mode=inference", f"ici_tensor_parallelism={infer_tp_degree}"],
        **base_kwargs,
    )

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    mesh_train = Mesh(maxtext_utils.create_device_mesh(cfg_train), cfg_train.mesh_axes)

    # `create_device_mesh` requires the ICI parallelism product to equal the
    # total visible device count, which does not hold for `infer_tp_degree`
    # (<=4). Build the mesh directly from a device slice instead.
    infer_device_slice = np.array(jax.devices()[:infer_tp_degree])
    infer_mesh_shape = tuple(infer_tp_degree if axis == "model" else 1 for axis in cfg_infer.mesh_axes)
    infer_devices = infer_device_slice.reshape(infer_mesh_shape)
    mesh_infer = Mesh(infer_devices, cfg_infer.mesh_axes)

    key = jax.random.PRNGKey(42)
    k_in, k_lyr = jax.random.split(key, 2)
    x_input = jax.random.normal(k_in, (batch_size, seq_len, emb_dim), dtype=jnp.float32)
    decoder_positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, 1))
    decoder_segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    x_input_np, decoder_positions_np, decoder_segment_ids_np = x_input, decoder_positions, decoder_segment_ids

    x_input = jax.device_put(x_input_np, NamedSharding(mesh_train, P(("data", "fsdp"), None, None)))
    decoder_positions = jax.device_put(decoder_positions_np, NamedSharding(mesh_train, P(("data", "fsdp"), None)))
    decoder_segment_ids = jax.device_put(decoder_segment_ids_np, NamedSharding(mesh_train, P(("data", "fsdp"), None)))

    infer_replicated_sharding = NamedSharding(mesh_infer, P())
    infer_x_input = jax.device_put(x_input_np, infer_replicated_sharding)
    infer_decoder_positions = jax.device_put(decoder_positions_np, infer_replicated_sharding)
    infer_decoder_segment_ids = jax.device_put(decoder_segment_ids_np, infer_replicated_sharding)

    from flax import nnx

    layer_train = qwen3_5.Qwen3_5DecoderLayer(
        config=cfg_train,
        mesh=mesh_train,
        layer_idx=0,
        model_mode="train",
        rngs=nnx.Rngs(params=10),
    )
    layer_infer = qwen3_5.Qwen3_5DecoderLayer(
        config=cfg_infer,
        mesh=mesh_infer,
        layer_idx=0,
        model_mode="prefill",
        rngs=nnx.Rngs(params=20),
    )

    sync_qwen3_5_layer_weights(layer_train, layer_infer)

    # `sync_qwen3_5_layer_weights` aliases Variable objects between the two
    # layers (raw attribute assignment). `nnx.split`/`nnx.merge` rebuilds
    # `layer_infer` with brand-new Variable objects wrapping device-placed
    # values, breaking that aliasing before the mesh-mismatched forward pass
    # (see tests/run_qwen3_5_layer_dump.py for the same fix + rationale).
    _infer_graphdef, _infer_state = nnx.split(layer_infer)
    _infer_state = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, infer_replicated_sharding), _infer_state
    )
    layer_infer = nnx.merge(_infer_graphdef, _infer_state)

    # 1. Full Cascaded Layer Run
    with jax.set_mesh(mesh_train):
        _, t_train = capture_qwen3_5_layer_intermediates(
            layer_train, x_input, decoder_segment_ids, decoder_positions, "train"
        )
    with jax.set_mesh(mesh_infer):
        _, t_infer = capture_qwen3_5_layer_intermediates(
            layer_infer, infer_x_input, infer_decoder_segment_ids, infer_decoder_positions, "prefill"
        )

    m_t16 = compute_metrics(t_train["T16_post_attn_layernorm_out"], t_infer["T16_post_attn_layernorm_out"])
    m_t19 = compute_metrics(t_train["T19_shared_expert_mlp_out"], t_infer["T19_shared_expert_mlp_out"])
    m_t20 = compute_metrics(t_train["T20_router_gate_logits"], t_infer["T20_router_gate_logits"])

    print("=" * 80)
    print("1. CASCADED FULL LAYER EXECUTION (Downstream of Attention):")
    print("=" * 80)
    print(f"  T16 (Post-Attn Norm Out) : L_inf={m_t16['max_abs_err']:.6e}, MAE={m_t16['mae']:.6e}")
    print(f"  T19 (Shared Expert Out)  : L_inf={m_t19['max_abs_err']:.6e}, MAE={m_t19['mae']:.6e}  (Amplification: {m_t19['max_abs_err']/m_t16['max_abs_err']:.2f}x)")
    print(f"  T20 (Router Gate Logits) : L_inf={m_t20['max_abs_err']:.6e}, MAE={m_t20['mae']:.6e}  (Amplification: {m_t20['max_abs_err']/m_t16['max_abs_err']:.2f}x)")

    # 2. Isolated Direct Test with Identical Clean Input
    clean_norm_input = t_train["T16_post_attn_layernorm_out"]
    # `layer_infer` now lives on `mesh_infer` (<=4 devices); re-place the
    # clean input (captured on `mesh_train`, 4 devices) before feeding it to
    # `layer_infer`'s submodules directly.
    infer_clean_norm_input = jax.device_put(clean_norm_input, infer_replicated_sharding)

    # Run Shared Expert on identical input
    with jax.set_mesh(mesh_train):
        shared_out_train_clean = layer_train.mlp.shared_expert(clean_norm_input, deterministic=True)
        gate_out_train_clean, _ = layer_train.mlp.routed_experts.gate(clean_norm_input)
        routed_out_train_clean, _, _ = layer_train.mlp.routed_experts(clean_norm_input)
    with jax.set_mesh(mesh_infer):
        shared_out_infer_clean = layer_infer.mlp.shared_expert(infer_clean_norm_input, deterministic=True)
        gate_out_infer_clean, _ = layer_infer.mlp.routed_experts.gate(infer_clean_norm_input)
        routed_out_infer_clean, _, _ = layer_infer.mlp.routed_experts(infer_clean_norm_input)

    m_t19_clean = compute_metrics(shared_out_train_clean, shared_out_infer_clean)
    m_t20_clean = compute_metrics(gate_out_train_clean, gate_out_infer_clean)
    m_t23_clean = compute_metrics(routed_out_train_clean, routed_out_infer_clean)

    print("\n" + "=" * 80)
    print("2. ISOLATED SUB-BLOCK EXECUTION (Identical Clean Input X):")
    print("=" * 80)
    print(f"  T19 (Shared Expert Intrinsic) : L_inf={m_t19_clean['max_abs_err']:.6e}, MAE={m_t19_clean['mae']:.6e}, CosSim={m_t19_clean['cos_sim']:.6f}")
    print(f"  T20 (Router Gate Intrinsic)   : L_inf={m_t20_clean['max_abs_err']:.6e}, MAE={m_t20_clean['mae']:.6e}, CosSim={m_t20_clean['cos_sim']:.6f}")
    print(f"  T23 (Routed MoE Intrinsic)    : L_inf={m_t23_clean['max_abs_err']:.6e}, MAE={m_t23_clean['mae']:.6e}, CosSim={m_t23_clean['cos_sim']:.6f}")
    print("=" * 80)


def main():
    print("[Local TPU VM] Running directly on locally-attached TPU chips.")
    run_isolation_diagnostics()


if __name__ == "__main__":
    main()
