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

"""Runner: Full Qwen3.5 model TRAINING-path vs INFERENCE-path LOGIT parity.

This is the core deliverable for the training/inference kernel parity
investigation: it runs the *full* Qwen3.5 model (token embedding -> N decoder
layers -> final norm -> lm_head) end-to-end through two execution paths and
compares the final LOGITS (not just intermediate tensors):

  * TRAINING path: `maxtext.models.models.Transformer` with
    `attention="flash"` (Tokamax Splash Attention) + Tokamax GMM v2 sparse
    MoE (`megablox=True, use_tokamax_gmm=True, use_gmm_v2=True,
    sparse_matmul=True`), `model_mode=MODEL_MODE_TRAIN`.
  * INFERENCE path: the same `Transformer` class with
    `attention="vllm_rpa"` (the default/plain tpu_inference Ragged
    Paged Attention Pallas v3 kernel, as actually served by vLLM) +
    `prefuse_moe_weights=True` (routes MoE through
    `RoutedMoE.fused_moe_matmul`, which calls
    `tpu_inference.layers.common.fused_moe_gmm.fused_moe_func` -- the real
    vLLM/tpu-inference Pallas MoE kernel), `model_mode=MODEL_MODE_PREFILL`.

Both models are constructed with identical `rngs=nnx.Rngs(params=SEED)` and
then their parameters are additionally force-synchronized via
`nnx.state`/`nnx.update` so any residual divergence is attributable to the
kernels, not to different random initialization.

No CPU mocks and no synthetic/reimplemented attention or MoE math are used on
either side -- both paths call the real production kernels used by MaxText
training and by vLLM serving, respectively.

Metrics reported on the final logits tensor [batch, seq_len, vocab_size]:
  * L_inf (max absolute error)
  * MAE (mean absolute error)
  * Cosine similarity (flattened)
  * Top-1 argmax agreement rate across positions (the metric that matters for
    greedy-decoding generation-quality parity)
  * Top-5 agreement rate (overlap between top-5 token sets per position)
  * KL divergence of the softmax distributions (train -> infer), averaged
    across positions

Runs directly on the locally-attached Cloud TPU v5p chips on this TPU VM. If
the run fails for any reason, this script prints the exact exception and
writes "NOT EXECUTED" (with the error) into the results doc -- it never
fabricates numbers.
"""

import os
import sys
import time
import traceback
from typing import Any

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from jax._src.pallas.mosaic import lowering as _mosaic_lowering  # noqa: F401
except Exception:  # pylint: disable=broad-except
    pass

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def compute_logit_parity_metrics(logits_train: jax.Array, logits_infer: jax.Array) -> dict[str, Any]:
    """Computes tensor-distance AND generation-quality parity metrics between two logit tensors.

    Args:
      logits_train: [batch, seq_len, vocab_size] logits from the training path.
      logits_infer: [batch, seq_len, vocab_size] logits from the inference path.

    Returns:
      Dict with L_inf, MAE, cosine similarity, top-1/top-5 argmax agreement
      rate, and mean KL divergence of the softmax distributions.
    """
    a = np.array(jax.device_get(logits_train), dtype=np.float32)
    b = np.array(jax.device_get(logits_infer), dtype=np.float32)
    assert a.shape == b.shape, f"Logit shape mismatch: {a.shape} vs {b.shape}"

    abs_diff = np.abs(a - b)
    max_abs_err = float(np.max(abs_diff))
    mae = float(np.mean(abs_diff))

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    cos_sim = float(np.dot(a_flat, b_flat) / (norm_a * norm_b + 1e-12))

    # Flatten batch & seq_len into "positions" for per-token argmax/KL metrics.
    batch, seq_len, vocab = a.shape
    a2 = a.reshape(batch * seq_len, vocab)
    b2 = b.reshape(batch * seq_len, vocab)

    top1_a = np.argmax(a2, axis=-1)
    top1_b = np.argmax(b2, axis=-1)
    top1_agreement = float(np.mean(top1_a == top1_b))

    k = min(5, vocab)
    top5_a = np.argsort(-a2, axis=-1)[:, :k]
    top5_b = np.argsort(-b2, axis=-1)[:, :k]
    top5_overlap = np.array(
        [len(set(top5_a[i]) & set(top5_b[i])) / k for i in range(a2.shape[0])]
    )
    top5_agreement = float(np.mean(top5_overlap))

    def _log_softmax(x):
        x = x - np.max(x, axis=-1, keepdims=True)
        log_z = np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
        return x - log_z

    log_p = _log_softmax(a2)  # training distribution
    log_q = _log_softmax(b2)  # inference distribution
    p = np.exp(log_p)
    kl_per_position = np.sum(p * (log_p - log_q), axis=-1)
    mean_kl = float(np.mean(kl_per_position))
    max_kl = float(np.max(kl_per_position))

    return {
        "shape": [int(batch), int(seq_len), int(vocab)],
        "max_abs_err": max_abs_err,
        "mae": mae,
        "cos_sim": cos_sim,
        "top1_argmax_agreement": top1_agreement,
        "top5_agreement": top5_agreement,
        "mean_kl_train_to_infer": mean_kl,
        "max_kl_train_to_infer": max_kl,
    }


def _build_rpa_attention_metadata_and_kv_caches(
    cfg,
    batch_size: int,
    seq_len: int,
    dtype,
    block_size: int = 128,
):
    """Builds the real `tpu_inference` RPA attention_metadata + per-layer KV
    caches needed to drive `Transformer.__call__` through the real Pallas RPA
    kernel for a full-model prefill pass (mirrors the pattern used inline in
    `tests/unit/qwen3_5_layer_dump_test.py::capture_qwen3_5_layer_intermediates`
    and `src/maxtext/models/models.py::Transformer.__init__`'s dummy metadata).
    """
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_pages = batch_size * num_blocks_per_seq

    try:
        from tpu_inference.layers.common.attention_metadata import AttentionMetadata

        # `query_start_loc` is (max_num_seqs + 1,): cumulative token offsets
        # per request, e.g. [0, seq_len, 2*seq_len, ...]. `request_distribution`
        # is a single global (3,) vector [num_decodes, num_prefills, num_mixed]
        # -- both must NOT be tiled per-batch, or the RPA kernel derives wrong
        # page/block indices and issues out-of-bounds DMAs.
        query_start_loc = jnp.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=jnp.int32)
        # Real semantics (tpu_inference/runner/tpu_runner.py):
        # [num_decode_requests, num_decode_requests, num_total_requests] --
        # NOT [decode, prefill, mixed] as might be guessed from the field
        # name. All `batch_size` requests here are prefill (0 decodes).
        request_distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)
        attention_metadata = AttentionMetadata(
            input_positions=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0).reshape(-1),
            block_tables=jnp.arange(total_pages, dtype=jnp.int32),
            seq_lens=jnp.array([seq_len] * batch_size, dtype=jnp.int32),
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )
    except ImportError:
        # Fall back to a duck-typed object matching the same field names, in
        # case the installed tpu_inference version's AttentionMetadata is a
        # strict dataclass with different required fields.
        class SimpleAttentionMetadata:  # pylint: disable=too-few-public-methods
            def __init__(self, input_positions, block_tables, seq_lens, query_start_loc, request_distribution):
                self.input_positions = input_positions
                self.block_tables = block_tables
                self.seq_lens = seq_lens
                self.query_start_loc = query_start_loc
                self.request_distribution = request_distribution

        attention_metadata = SimpleAttentionMetadata(
            input_positions=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0).reshape(-1),
            block_tables=jnp.arange(total_pages, dtype=jnp.int32),
            seq_lens=jnp.array([seq_len] * batch_size, dtype=jnp.int32),
            query_start_loc=jnp.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=jnp.int32),
            request_distribution=jnp.array([0, batch_size, 0], dtype=jnp.int32),
        )

    num_kv_heads = cfg.num_kv_heads
    head_dim = cfg.head_dim
    try:
        from tpu_inference.layers.common.attention_interface import get_kv_cache_shape

        kv_shape = get_kv_cache_shape(total_pages, block_size, num_kv_heads, head_dim, dtype)
    except Exception:  # pylint: disable=broad-except
        kv_shape = (total_pages, block_size, num_kv_heads, 2, head_dim)

    kv_caches = [jnp.zeros(kv_shape, dtype=dtype) for _ in range(cfg.num_decoder_layers)]
    return attention_metadata, kv_caches


def run_full_model_logit_parity(
    dtype_str: str,
    batch_size: int = 2,
    seq_len: int = 128,
    num_decoder_layers: int = 2,
    emb_dim: int = 2048,
    moe_mlp_dim: int = 512,
    num_experts: int = 8,
    num_experts_per_tok: int = 8,
    vocab_size: int = 32000,
) -> dict[str, Any]:
    """Runs the full Qwen3.5 model through the TRAINING and INFERENCE paths on
    real TPU hardware and returns the final-logits parity metrics.
    """
    print(f"\n>>> Running Qwen3.5 FULL-MODEL logit parity [{dtype_str}, {num_decoder_layers} layers] on TPU...")

    base_kwargs = {
        "override_model_config": True,
        # `num_decoder_layers` is a DERIVED field
        # (`self.num_decoder_layers = (2**layer_scale) * self.base_num_decoder_layers`,
        # src/maxtext/configs/types.py), recomputed post-init and silently
        # overwriting any `num_decoder_layers=N` kwarg passed here. The real
        # depth control is `base_num_decoder_layers` (`layer_scale` defaults
        # to 0 via `global_parameter_scale=1`, so `2**0=1` -- no need to set
        # it explicitly, and it isn't a real settable Field anyway).
        "base_num_decoder_layers": num_decoder_layers,
        "model_name": "qwen3.5-35b-a3b",
        "base_emb_dim": emb_dim,
        "base_mlp_dim": moe_mlp_dim,
        "base_moe_mlp_dim": moe_mlp_dim,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "vocab_size": vocab_size,
        "max_target_length": seq_len,
        "max_prefill_predict_length": seq_len,
        "per_device_batch_size": 1.0,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
        "scan_layers": False,
        "enable_checkpointing": False,
        "log_config": False,
        "inhomogeneous_layer_cycle_interval": 1,
        "norm_topk_prob": True,
        "float32_logits": True,
        "float32_gate_logits": True,
        "float32_weight_sum": True,
    }

    train_kwargs = dict(base_kwargs)
    train_kwargs.update({
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "sparse_matmul": True,
        "wi_tile_fwd_batch_seq": 256,
        "wi_tile_fwd_embed_dim": 128,
        "wi_tile_fwd_mlp_dim": 128,
        "use_tokamax_splash": True,
        "sa_use_base2_exp": False,
        "sa_fuse_reciprocal": True,
    })

    cfg_train = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path(),
            "attention=flash",
            "use_tokamax_splash=True",
            "sa_use_base2_exp=False",
            "sa_fuse_reciprocal=True",
            "sparse_matmul=True",
            "megablox=True",
            "use_tokamax_gmm=True",
            "use_gmm_v2=True",
        ],
        weight_dtype=dtype_str,
        dtype=dtype_str,
        **train_kwargs,
    )

    # Tensor-parallel degree for the inference mesh is capped at
    # `num_kv_heads`: the RPA kernel shards the KV cache's head axis across
    # the ('model', 'expert', 'dcp') mesh axes, and that combined axis size
    # must evenly divide num_kv_heads (2 for qwen3.5-35b-a3b).
    infer_tp_degree = min(len(jax.devices()), cfg_train.num_kv_heads)

    cfg_infer = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path("inference/vllm.yml"),
            "attention=vllm_rpa",
            "prefuse_moe_weights=True",
            "model_call_mode=inference",
            # Tensor-parallel sharding (the `model` mesh axis), as real vLLM
            # TPU serving does -- NOT data-parallel (`data` axis must stay
            # size 1: `AttentionMetadata.request_distribution` is a
            # fixed-shape (3,) global vector that cannot be evenly sharded
            # across >1 "data" replicas).
            f"ici_tensor_parallelism={infer_tp_degree}",
        ],
        weight_dtype=dtype_str,
        dtype=dtype_str,
        **base_kwargs,
    )

    train_devices = maxtext_utils.create_device_mesh(cfg_train)
    train_mesh = Mesh(train_devices, cfg_train.mesh_axes)

    # `create_device_mesh` requires the ICI parallelism product to equal the
    # total visible device count. Here it must instead equal `infer_tp_degree`
    # (<=4, capped by num_kv_heads): every one of the 7 mesh axes participates
    # in one of the two `_ragged_paged_attention` shard_map divisibility
    # groups (('data','pcp','attn_dp','attn_dp_expert') for per-request
    # metadata, ('model','expert','dcp') for the KV-head dim), so there is no
    # "free" axis to soak up leftover devices without breaking one of them.
    # Build the mesh directly from a device slice instead.
    infer_device_slice = np.array(jax.devices()[:infer_tp_degree])
    infer_mesh_shape = tuple(infer_tp_degree if axis == "model" else 1 for axis in cfg_infer.mesh_axes)
    infer_devices = infer_device_slice.reshape(infer_mesh_shape)
    infer_mesh = Mesh(infer_devices, cfg_infer.mesh_axes)

    seed = 42
    train_model = models.Transformer(
        config=cfg_train, mesh=train_mesh, quant=None, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=seed)
    )
    infer_model = models.Transformer(
        config=cfg_infer, mesh=infer_mesh, quant=None, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=seed)
    )

    # Force bit-identical weights across both paths (belt-and-suspenders on
    # top of the identical RNG seed) so any divergence in the final logits is
    # attributable purely to kernel numerics, not initialization drift.
    print("  -> Synchronizing full-model parameters (nnx.state / nnx.update)...")
    train_param_state = nnx.state(train_model, nnx.Param)
    # `train_param_state`'s leaves are placed on `train_mesh` (4 devices).
    # `infer_mesh` only spans `infer_tp_degree` (<=4) devices, so every leaf
    # must be re-placed (fully replicated -- correctness test, not a
    # performance benchmark) onto `infer_mesh` before `nnx.update`, or the
    # infer forward pass raises a device-mismatch error.
    infer_replicated_sharding = NamedSharding(infer_mesh, P())
    train_param_state_for_infer = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, infer_replicated_sharding), train_param_state
    )
    nnx.update(infer_model, train_param_state_for_infer)

    dtype_jax = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32
    key = jax.random.PRNGKey(7)
    k_tok, _ = jax.random.split(key)
    token_ids_np = jax.random.randint(k_tok, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32)
    decoder_positions_np = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
    decoder_segment_ids_np = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    token_ids = jax.device_put(token_ids_np, NamedSharding(train_mesh, P(("data", "fsdp"), None)))
    decoder_positions = jax.device_put(decoder_positions_np, NamedSharding(train_mesh, P(("data", "fsdp"), None)))
    decoder_segment_ids = jax.device_put(decoder_segment_ids_np, NamedSharding(train_mesh, P(("data", "fsdp"), None)))

    # Separate copies placed on `infer_mesh` for the inference forward pass.
    infer_token_ids = jax.device_put(token_ids_np, infer_replicated_sharding)
    infer_decoder_positions = jax.device_put(decoder_positions_np, infer_replicated_sharding)
    infer_decoder_segment_ids = jax.device_put(decoder_segment_ids_np, infer_replicated_sharding)

    print("  -> Executing TRAINING forward pass (Tokamax Splash Attention + Tokamax GMM v2 MoE)...")
    logits_train = train_model(
        token_ids,
        decoder_positions,
        decoder_segment_ids,
        model_mode=MODEL_MODE_TRAIN,
    )
    logits_train = jax.block_until_ready(logits_train)

    print("  -> Building real vLLM batched-RPA attention_metadata + per-layer KV caches...")
    attention_metadata, kv_caches = _build_rpa_attention_metadata_and_kv_caches(
        cfg_infer, batch_size, seq_len, dtype_jax
    )
    attention_metadata = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, infer_replicated_sharding), attention_metadata
    )
    kv_caches = [jax.device_put(kv, infer_replicated_sharding) for kv in kv_caches]

    print("  -> Executing INFERENCE forward pass (vLLM RPA (default Pallas RPA v3) + vLLM Pallas Fused MoE)...")
    # The IFRT proxy connection to the SPS Pathways head has been observed to
    # drop mid-run on long-lived jobs (see docs/train_infer_logit_parity.md).
    # The training-path call above is cheap to redo, so retry the inference
    # call a couple of times against the *same* live connection in case the
    # disconnect is a transient per-call blip rather than a dead session.
    last_err: Exception | None = None
    logits_infer = None
    for attempt in range(3):
        try:
            # For `attention in ("vllm_rpa", "vllm_batched_rpa")`,
            # `Transformer.__call__` returns `(hidden_state, kv_caches)`
            # rather than logits -- in real vLLM serving, logits are computed
            # separately by the vLLM model-runner's own head. To keep this an
            # apples-to-apples comparison of the ATTENTION/MoE kernels (not a
            # reimplementation of vLLM's head), project the returned hidden
            # state to logits using the same `apply_output_head` function the
            # training path uses internally.
            hidden_state_infer, _ = infer_model(
                infer_token_ids,
                infer_decoder_positions,
                infer_decoder_segment_ids,
                model_mode=MODEL_MODE_PREFILL,
                attention_metadata=attention_metadata,
                kv_caches=kv_caches,
            )
            logits_infer = infer_model.decoder.apply_output_head(
                infer_model.token_embedder, hidden_state_infer, deterministic=True, model_mode=MODEL_MODE_PREFILL
            )
            logits_infer = jax.block_until_ready(logits_infer)
            last_err = None
            break
        except Exception as e:  # pylint: disable=broad-except
            last_err = e
            print(f"     [retry {attempt + 1}/3] inference forward pass failed: {e}")
            time.sleep(5)
    if last_err is not None:
        raise last_err

    print("  -> Computing final-logit parity metrics...")
    metrics = compute_logit_parity_metrics(logits_train, logits_infer)
    print(
        f"     L_inf={metrics['max_abs_err']:.6e}  MAE={metrics['mae']:.6e}  CosSim={metrics['cos_sim']:.6f}\n"
        f"     Top1 Agreement={metrics['top1_argmax_agreement']:.4%}  Top5 Agreement={metrics['top5_agreement']:.4%}\n"
        f"     Mean KL(train||infer)={metrics['mean_kl_train_to_infer']:.6e}  Max KL={metrics['max_kl_train_to_infer']:.6e}"
    )

    import gc

    del train_model, infer_model, logits_train, logits_infer
    gc.collect()
    return metrics


def main():
    # This script runs directly on a TPU VM with locally-attached chips.
    results_doc_path = os.path.join(os.getcwd(), "docs", "train_infer_logit_parity.md")
    os.makedirs(os.path.dirname(results_doc_path), exist_ok=True)

    all_metrics: dict[str, Any] = {}
    error_info: str | None = None

    def _run_all():
        print(f"  JAX Platforms: {jax.config.jax_platforms}")
        print(f"  Detected TPU Devices ({len(jax.devices())}): {jax.devices()}\n")
        # batch_size must be divisible by the number of TPU devices in the
        # data/fsdp mesh axes (4 on a v5p 2x2x1 slice).
        batch_size = max(len(jax.devices()), 4)
        for num_layers in [1, 2, 40]:
            for dtype_str in ["bfloat16", "float32"]:
                all_metrics[(dtype_str, num_layers)] = run_full_model_logit_parity(
                    dtype_str=dtype_str,
                    batch_size=batch_size,
                    seq_len=128,
                    num_decoder_layers=num_layers,
                )

    try:
        print("=" * 80)
        print("[Local TPU VM] Running directly on locally-attached TPU chips (no SPS proxy).")
        print("=" * 80)
        _run_all()
    except Exception:  # pylint: disable=broad-except
        error_info = traceback.format_exc()
        print("\n[FAILED] TPU run did not complete successfully.")
        print(error_info)

    time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    if error_info is None and all_metrics:
        rows = []
        for (dtype_str, num_layers), m in all_metrics.items():
            rows.append(
                f"| {num_layers} | {dtype_str} | {m['shape']} | {m['max_abs_err']:.3e} | {m['mae']:.3e} | "
                f"{m['cos_sim']:.6f} | {m['top1_argmax_agreement']:.4%} | {m['top5_agreement']:.4%} | "
                f"{m['mean_kl_train_to_infer']:.3e} | {m['max_kl_train_to_infer']:.3e} |"
            )
        status_block = (
            "**Status: EXECUTED on local TPU VM (locally-attached chips).**\n\n"
            "| Layers | DType | Shape [B,S,V] | $L_\\infty$ | MAE | CosSim | Top-1 Agreement | "
            "Top-5 Agreement | Mean KL(train‖infer) | Max KL |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n" + "\n".join(rows)
        )
    else:
        status_block = (
            "**Status: NOT EXECUTED.** The local TPU VM run did not complete. Exact error:\n\n"
            f"```\n{error_info or 'Unknown error (no exception captured).'}\n```\n"
        )

    platform_str = "Google Cloud TPU v5p, local TPU VM (locally-attached chips, no SPS proxy)"
    doc_content = f"""# Train vs. Inference Final-Logit Parity (Qwen3.5)

**Date / Timestamp:** {time_str}
**Hardware Platform:** {platform_str}
**Script:** `tests/run_qwen3_5_logit_parity.py`

## Methodology

Runs the **full** Qwen3.5 model (`maxtext.models.models.Transformer`:
token embedding -> N decoder layers -> final RMSNorm -> lm_head) end-to-end
on identical random token-id input, through two paths, and compares the
final **logits** tensor `[batch, seq_len, vocab_size]` -- not intermediate
activations.

* **Training path:** `attention="flash"` (Tokamax Splash Attention),
  `megablox=True, use_tokamax_gmm=True, use_gmm_v2=True, sparse_matmul=True`
  (Tokamax GMM v2 MoE), `model_mode=MODEL_MODE_TRAIN`.
* **Inference path:** `attention="vllm_rpa"` (default Pallas RPA v3; the real
  `tpu_inference` Ragged Paged Attention Pallas kernel, as served by
  vLLM), `prefuse_moe_weights=True` (routes MoE through
  `RoutedMoE.fused_moe_matmul` -> `tpu_inference.layers.common.fused_moe_gmm.fused_moe_func`,
  vLLM's real Pallas MoE kernel), `model_mode=MODEL_MODE_PREFILL`.

Both models are constructed with the same `nnx.Rngs(params=42)` seed, and
weights are additionally force-synchronized via `nnx.state(train_model,
nnx.Param)` / `nnx.update(infer_model, ...)` so residual differences are
attributable to kernel numerics, not initialization. No CPU mocks and no
reimplemented attention/MoE math are used on either side.

Metrics computed on the final logits:
* **L_inf / MAE / Cosine similarity** -- raw tensor-distance metrics.
* **Top-1 argmax agreement rate** -- fraction of positions where
  `argmax(logits_train) == argmax(logits_infer)`. This is what greedy
  decoding parity actually depends on.
* **Top-5 agreement rate** -- average overlap between the top-5 token sets.
* **KL divergence (train‖infer)**, mean and max across positions -- how much
  the sampling distributions actually diverge.

## Results

{status_block}

## Learnings

This section consolidates everything learned across the whole kernel-parity
investigation on this branch (previously spread across `docs/learnings.md`,
`docs/parity_improvement_story.md`, and `docs/next_plan.md`, which have been
folded into this document and removed to avoid stale duplicates). All numbers
below were produced by the standalone kernel repro scripts
(`tests/run_attention_kernel_repro.py`,
`tests/run_attention_batched_rpa_repro.py`,
`tests/run_moe_kernel_repro.py`) and the 1-layer intermediate-tensor dump
(`tests/run_qwen3_5_layer_dump.py`, results in
`docs/qwen3_5_kernel_drift_results.md`), all executed directly on the local
TPU VM's locally-attached Cloud TPU v5p chips. They describe
**intermediate-tensor** parity; the
final-logit numbers in the Results section above (or the "NOT EXECUTED"
status) are the authoritative full-model parity numbers for this document.

### 1. Attention: Splash (training) vs. RPA (inference)

* In FP32, isolated Splash Attention vs. RPA has $L_\\infty \\approx 1.53
  \\times 10^{{-5}}$, CosSim $\\approx 0.999999$.
* In BF16, both kernels show $L_\\infty \\approx 1.56 \\times 10^{{-2}}$
  against an exact FP32 math reference -- this is the **1-ULP quantization
  floor** of BF16's 7-bit mantissa for values in $[2.0, 4.0)$, not a kernel
  bug. Over 60% of output elements are bit-identical.
* Setting `sa_use_base2_exp=False` (native base-$e$ exponential, matching
  RPA and the exact reference, instead of Tokamax Splash's default base-2
  fast-exp) reduced attention-core MAE by ~10.8% and MSE by ~16.1% vs. the
  `sa_use_base2_exp=True` default. This fix (`sa_use_base2_exp=False,
  sa_fuse_reciprocal=True`) is applied on the training side throughout this
  investigation.

### 2. MoE: Tokamax GMM v2 (training) vs. vLLM Fused MoE (inference)

* In isolation (identical input activations, FP32), both kernels agree with
  the exact FP32 math reference to $L_\\infty \\approx 10^{{-8}}$-$10^{{-5}}$
  and CosSim = 1.000000 -- i.e. **the MoE kernels themselves have no
  meaningful numerical divergence.**
* Aligning the training-side GMM contraction tile size to match inference's
  auto-tiling (`wi_tile_fwd_batch_seq=256, wi_tile_fwd_embed_dim=128,
  wi_tile_fwd_mlp_dim=128`) reduced training-vs-inference MoE $L_\\infty$
  from $3.32\\times10^{{-5}}$ to $2.98\\times10^{{-8}}$ (FP32).
* `float32_gate_logits=True` and `float32_weight_sum=True` keep router
  logits and the top-$K$ weighted combination in FP32, preventing
  boundary-token misrouting and rounding loss in the expert combination.

### 3. Error amplification through the MoE MLP (why 1-layer FP32 error was ~7e-3, not ~1e-5)

* A 1-layer full decoder (attention + MoE) end-to-end FP32 comparison showed
  $L_\\infty \\approx 7.12\\times10^{{-3}}$ at the layer output, even though
  both kernels are individually near machine precision in isolation.
* Diagnosed via `tests/diagnose_t19_t20_amplification.py`, which compares
  the **cascaded** (real, error-compounding) execution against an
  **isolated** execution where both training and inference MoE sub-blocks
  are fed the *identical* clean post-attention-norm activation. The isolated
  run reproduces the machine-precision agreement from Learning 2, confirming
  the $7\\times10^{{-3}}$ error is not intrinsic to the MoE kernel -- it is
  the small attention-core residual ($\\Delta x \\approx 1.53\\times10^{{-5}}$)
  passed through 3 successive linear projections
  ($W_{{gate}}, W_{{up}}, W_{{down}}$) in the MoE MLP, whose combined spectral
  norm ($\\sim 10^2$-$10^3$) amplifies it: $\\Delta y \\approx \\|W_{{gate}}\\|
  \\cdot \\|W_{{up}}\\| \\cdot \\|W_{{down}}\\| \\cdot \\Delta x$.
* Practical takeaway: don't chase intermediate-tensor $L_\\infty$ deltas at
  the MoE block in isolation from what feeds it -- verify the *source*
  (attention) delta and treat downstream amplification as expected linear
  algebra, not a new bug. The metrics that matter for actual generation
  quality are the final-logit top-1/top-5 argmax agreement and KL divergence
  reported in the Results section above, since RMSNorm variance-reset and
  the $O(1/\\sqrt{{L}})$ residual-stream relative-error decay in a full
  multi-layer Pre-LN stack are expected to keep this bounded rather than
  exploding across layers -- **this has not yet been verified empirically
  beyond a 1-2 layer stack on this branch; a depth-scaling sweep (1, 2, 4, 8+
  layers) remains open future work.**

### 4. What is verified vs. still open

Verified on real Cloud TPU v5p hardware (local TPU VM) with real kernels
(no mocks):
* Standalone attention kernel parity (Splash vs. default RPA vs. batched
  RPA vs. exact reference), FP32 and BF16.
* Standalone MoE kernel parity (Tokamax GMM v2 vs. vLLM fused MoE vs. exact
  reference), FP32.
* 1-decoder-layer (attention + MoE) intermediate-tensor parity, FP32 and
  BF16, 25-tensor breakdown (`docs/qwen3_5_kernel_drift_results.md`).
* Error-amplification root-cause diagnosis (isolated vs. cascaded MoE
  sub-block execution).

Still open / not yet verified on this branch:
* Full-model, multi-layer (>2 layer) logit parity at production depth
  (32-64+ layers) -- only the 1-2 layer results in this document exist so
  far; run this script with a larger `num_decoder_layers` to extend.
* Autoregressive multi-step generation / top-1 greedy-token-match parity
  across a full decode rollout (KV cache reuse across steps), as opposed to
  a single prefill forward pass.
* Real (non-random) token inputs / real checkpoint weights, as opposed to
  freshly-initialized random weights synchronized between the two paths.

## Standalone Diagnostic Scripts

| Script | Purpose |
| :--- | :--- |
| `tests/run_qwen3_5_logit_parity.py` | **This document's source.** Full-model training-path vs. inference-path final logit parity. |
| `tests/run_attention_kernel_repro.py` | Multi-config sweep: Splash (legacy JAX / Tokamax variants) vs. default RPA vs. batched RPA vs. exact reference. |
| `tests/run_attention_batched_rpa_repro.py` | Focused Splash vs. default-RPA-v3 vs. batched-RPA 3-way comparison. |
| `tests/run_moe_kernel_repro.py` | Multi-config sweep: Tokamax GMM v2 (various tile sizes) vs. legacy Megablox vs. dense-einsum reference vs. vLLM fused MoE. |
| `tests/run_qwen3_5_layer_dump.py` | 1-decoder-layer, 25-intermediate-tensor dump and drift comparison; writes `docs/qwen3_5_kernel_drift_results.md`. |
| `tests/diagnose_t19_t20_amplification.py` | Isolated-vs-cascaded MoE sub-block diagnostic explaining the 1-layer error amplification mechanism (Learning 3 above). |

All of the above run directly on the local TPU VM's locally-attached Cloud
TPU v5p chips (no remote proxy needed), and require the `vllm-tpu` /
`tpu_inference` packages for the real inference-side kernels. Run with:
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python NEW_MODEL_DESIGN=1 VLLM_TARGET_DEVICE=tpu \\
python3 tests/run_qwen3_5_logit_parity.py
```
"""

    with open(results_doc_path, "w", encoding="utf-8") as f:
        f.write(doc_content)
    print(f"\nResults written to {results_doc_path}")

    if error_info is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
