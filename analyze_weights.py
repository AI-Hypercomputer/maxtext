"""
Analyze layer-2 weights from GCS to check if MaxText->vLLM conversion is correct.

Run with:
  python analyze_weights.py --bucket mohit-maxtext-logs \
    --prefix debug-weights-0428-2/saved_weights

Checks:
  1. Shapes match between maxtext_ and vllm_ arrays
  2. Whether the conversion is a permutation/reshape of the same values (same set of
     unique values → correct math, wrong layout) vs genuinely different values
     (different checkpoint)
  3. For w13: checks whether gate and up chunks are correctly interleaved
  4. For w2: checks whether the double-transpose is consistent
  5. For qkv: checks multiple candidate layouts against vLLM's saved weight
"""

import argparse
import io
import numpy as np
from google.cloud import storage


def to_f32(arr):
    """Convert array to float32, handling bfloat16 stored as |V2 void dtype."""
    if arr.dtype.kind == 'V' and arr.dtype.itemsize == 2:
        # bfloat16 as raw 2-byte void: view as uint16 then shift to float32
        u32 = arr.view(np.uint16).astype(np.uint32) << 16
        return u32.view(np.float32)
    return arr.astype(np.float32)


def download(bucket, blob_name):
    buf = io.BytesIO()
    bucket.blob(blob_name).download_to_file(buf)
    buf.seek(0)
    return np.load(buf, allow_pickle=False)


def report(name, mt, vl):
    print(f"\n{'='*70}")
    print(f"KEY: {name}")
    print(f"  maxtext shape : {mt.shape}  dtype={mt.dtype}")
    print(f"  vllm    shape : {vl.shape}  dtype={vl.dtype}")

    if mt.shape != vl.shape:
        print("  *** SHAPE MISMATCH — cannot compare values ***")
        return

    mt_f = to_f32(mt)
    vl_f = to_f32(vl)

    diff = np.abs(mt_f - vl_f)
    print(f"  max_diff      : {diff.max():.6f}")
    print(f"  mean_diff     : {diff.mean():.6f}")
    print(f"  frac>1e-3     : {(diff > 1e-3).mean()*100:.2f}%")

    mt_sorted = np.sort(mt_f.ravel())
    vl_sorted = np.sort(vl_f.ravel())
    is_permutation = np.allclose(mt_sorted, vl_sorted, atol=1e-4)
    print(f"  same_values (permutation/reshape only): {is_permutation}")

    if not is_permutation:
        print("  >>> Values are genuinely different — likely different checkpoints or precision")
    else:
        print("  >>> Same values, different layout — CONVERSION BUG (wrong permutation)")


def analyze_qkv(mt, vl, num_q_heads, num_kv_heads, head_dim, tp):
    """
    mt and vl both have shape (total_qkv_dim, d_model) = (9216, 4096).

    bench_weight_sync.py produces:  [Q_first, K, Q_second, V] × tp  (TP-interleaved)
    vLLM QKVParallelLinear expects: [Q_all, K_all, V_all]  (plain concat)

    This function checks both layouts against the saved vllm weight.
    """
    print(f"\n  --- qkv layout analysis ---")
    print(f"  num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, tp={tp}")

    actual_tp = min(tp, num_kv_heads)
    q_per_tp = num_q_heads // actual_tp
    kv_per_tp = num_kv_heads // actual_tp
    total_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
    d_model = mt.shape[1] if mt.ndim == 2 else mt.shape[-1]

    mt_f = to_f32(mt)
    vl_f = to_f32(vl)

    # --- 1. Is mt a row-permutation of vl? ---
    mt_sorted = np.sort(mt_f.ravel())
    vl_sorted = np.sort(vl_f.ravel())
    is_perm = np.allclose(mt_sorted, vl_sorted, atol=1e-4)
    print(f"  same value multiset (permutation): {is_perm}")
    if not is_perm:
        print("  >>> NOT a permutation — values differ. Possible checkpoint mismatch.")
        return

    # --- 2. Decode the TP-interleaved layout that bench_weight_sync.py builds ---
    # Layout: for each tp rank r in [0..actual_tp-1]:
    #   rows [r*(q_per_tp+kv_per_tp*2)*head_dim .. ] =
    #     [Q_first(8 heads), K(kv_per_tp heads), Q_second(8 heads), V(kv_per_tp heads)]
    #   where Q_first = q_by_tp[:, r, :q_per_tp//2, :], etc.
    #
    # This was assembled as: qkv_by_tp reshape + transpose -> (total_dim, d_model)
    # Rows in mt = [q_first_tp0, k_tp0, q_second_tp0, v_tp0,
    #               q_first_tp1, k_tp1, q_second_tp1, v_tp1, ...]
    # where q_first_tp_r has q_per_tp//2 * head_dim rows, k/v have kv_per_tp*head_dim rows

    q_half = q_per_tp // 2 * head_dim
    kv_sz  = kv_per_tp * head_dim
    block  = (q_half + kv_sz + q_half + kv_sz)  # rows per TP rank in mt

    # Extract Q, K, V from mt (TP-interleaved)
    q_rows_mt = []
    k_rows_mt = []
    v_rows_mt = []
    for r in range(actual_tp):
        base = r * block
        q_rows_mt.append(mt_f[base : base + q_half])
        k_rows_mt.append(mt_f[base + q_half : base + q_half + kv_sz])
        q_rows_mt.append(mt_f[base + q_half + kv_sz : base + q_half + kv_sz + q_half])
        v_rows_mt.append(mt_f[base + q_half + kv_sz + q_half : base + block])

    q_mt = np.concatenate(q_rows_mt)  # (num_q_heads * head_dim, d_model)
    k_mt = np.concatenate(k_rows_mt)  # (num_kv_heads * head_dim, d_model)
    v_mt = np.concatenate(v_rows_mt)  # (num_kv_heads * head_dim, d_model)
    print(f"  mt decoded Q shape: {q_mt.shape}, K: {k_mt.shape}, V: {v_mt.shape}")

    # --- 3. Extract Q, K, V from vl (assume standard [Q_all, K_all, V_all]) ---
    q_sz = num_q_heads * head_dim
    kv_sz_full = num_kv_heads * head_dim
    q_vl = vl_f[:q_sz]
    k_vl = vl_f[q_sz : q_sz + kv_sz_full]
    v_vl = vl_f[q_sz + kv_sz_full :]
    print(f"  vl [Q,K,V] split Q: {q_vl.shape}, K: {k_vl.shape}, V: {v_vl.shape}")

    dq = np.abs(q_mt - q_vl)
    dk = np.abs(k_mt - k_vl)
    dv = np.abs(v_mt - v_vl)
    print(f"  Q diff (mt-interleaved vs vl-[Q,K,V]): max={dq.max():.6f}, mean={dq.mean():.6f}")
    print(f"  K diff: max={dk.max():.6f}, mean={dk.mean():.6f}")
    print(f"  V diff: max={dv.max():.6f}, mean={dv.mean():.6f}")

    if dq.max() < 1e-3 and dk.max() < 1e-3 and dv.max() < 1e-3:
        print("  >>> Layout hypothesis CONFIRMED: mt uses TP-interleaved, vl uses [Q,K,V].")
        print("  >>> FIX: bench_weight_sync.py should output [Q_all, K_all, V_all] not TP-interleaved.")
    else:
        print("  >>> Layout hypothesis did not match. Trying alternative decompositions...")
        # Try: vl uses TP-interleaved too (same layout as mt)
        diff_direct = np.abs(mt_f - vl_f)
        print(f"  Direct diff (same layout): max={diff_direct.max():.6f}")

        # Try: vl is [Q_tp0, K_tp0, V_tp0, Q_tp1, K_tp1, V_tp1, ...] (vLLM TP gather order)
        block_v = (q_per_tp + kv_per_tp * 2) * head_dim
        q_rows_vl2, k_rows_vl2, v_rows_vl2 = [], [], []
        for r in range(actual_tp):
            base = r * block_v
            q_rows_vl2.append(vl_f[base : base + q_per_tp * head_dim])
            k_rows_vl2.append(vl_f[base + q_per_tp * head_dim : base + q_per_tp * head_dim + kv_sz])
            v_rows_vl2.append(vl_f[base + q_per_tp * head_dim + kv_sz : base + block_v])
        q_vl2 = np.concatenate(q_rows_vl2)
        k_vl2 = np.concatenate(k_rows_vl2)
        v_vl2 = np.concatenate(v_rows_vl2)
        dq2 = np.abs(q_mt - q_vl2)
        dk2 = np.abs(k_mt - k_vl2)
        dv2 = np.abs(v_mt - v_vl2)
        print(f"  Q diff (mt-interleaved vs vl-TP-gather): max={dq2.max():.6f}")
        print(f"  K diff: max={dk2.max():.6f}")
        print(f"  V diff: max={dv2.max():.6f}")

    # --- 4. What would the corrected mt look like? ---
    mt_corrected = np.concatenate([q_mt, k_mt, v_mt], axis=0)
    diff_corrected = np.abs(mt_corrected - vl_f)
    print(f"\n  If mt used [Q_all,K_all,V_all] (corrected): max_diff={diff_corrected.max():.6f}")
    if diff_corrected.max() < 1e-3:
        print("  >>> CONFIRMED FIX: output [Q_all, K_all, V_all] instead of TP-interleaved.")


def analyze_w13(mt, vl, tp, num_experts):
    print(f"\n  --- w13 layout analysis (tp={tp}, E={num_experts}) ---")
    E = vl.shape[0]
    print(f"  vllm w13 shape: {vl.shape}")
    print(f"  maxtext w13 shape: {mt.shape}")

    mt_f = to_f32(mt)
    vl_f = to_f32(vl)

    if vl.ndim == 3:
        d_model = vl_f.shape[1]
        two_tp_padded_c = vl_f.shape[2]
        padded_c = two_tp_padded_c // (2 * tp)
        print(f"  d_model={d_model}, 2*tp*padded_c={two_tp_padded_c}, padded_c={padded_c}")

        vl_t = vl_f.transpose(0, 2, 1)
        vl_r = vl_t.reshape(E, tp, 2, padded_c, d_model)
        gate_chunks = vl_r[:, :, 0, :, :]
        up_chunks   = vl_r[:, :, 1, :, :]
        print(f"  gate_chunks shape: {gate_chunks.shape}")
        print(f"  up_chunks   shape: {up_chunks.shape}")

        if mt.ndim == 3:
            print(f"  maxtext value range: [{mt_f.min():.4f}, {mt_f.max():.4f}]")
            print(f"  gate   value range: [{gate_chunks.min():.4f}, {gate_chunks.max():.4f}]")
            print(f"  up     value range: [{up_chunks.min():.4f}, {up_chunks.max():.4f}]")


def analyze_w2(mt, vl):
    print(f"\n  --- w2 layout analysis ---")
    print(f"  maxtext shape: {mt.shape}")
    print(f"  vllm    shape: {vl.shape}")

    mt_f = to_f32(mt)
    vl_f = to_f32(vl)

    if mt.ndim == 3 and vl.ndim == 3:
        if mt.shape == (vl.shape[0], vl.shape[2], vl.shape[1]):
            diff = np.abs(mt_f.transpose(0, 2, 1) - vl_f)
            print(f"  max_diff after transpose(0,2,1): {diff.max():.6f}")
            if diff.max() < 1e-3:
                print("  >>> w2 transpose is CORRECT")
            else:
                print("  >>> w2 transpose gives wrong values — checkpoint diff or bug")
        else:
            print(f"  shapes incompatible for simple transpose check")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="mohit-maxtext-logs")
    parser.add_argument("--prefix", default="debug-weights-0428-2/saved_weights")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--num_q_heads", type=int, default=64)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=128)
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    L = args.layer

    keys = [
        "mlp_experts_w13_weight",
        "mlp_experts_w2_weight",
        "mlp_gate_weight",
        "self_attn_qkv_proj_weight",
        "self_attn_o_proj_weight",
        "self_attn_q_norm_weight",
        "self_attn_k_norm_weight",
        "input_layernorm_weight",
        "post_attention_layernorm_weight",
    ]

    arrays = {}
    for k in keys:
        for prefix in ["maxtext", "vllm"]:
            blob_name = f"{args.prefix}/{prefix}_vllm_model_model_layers_{L}_{k}.npy"
            label = f"{prefix}_{k}"
            try:
                arrays[label] = download(bucket, blob_name)
                print(f"Loaded {label}: shape={arrays[label].shape} dtype={arrays[label].dtype}")
            except Exception as e:
                print(f"Failed to load {label}: {e}")

    print("\n" + "="*70)
    print("SUMMARY COMPARISON")

    for k in keys:
        mt_key = f"maxtext_{k}"
        vl_key = f"vllm_{k}"
        if mt_key in arrays and vl_key in arrays:
            report(k, arrays[mt_key], arrays[vl_key])

    if "maxtext_self_attn_qkv_proj_weight" in arrays and "vllm_self_attn_qkv_proj_weight" in arrays:
        analyze_qkv(
            arrays["maxtext_self_attn_qkv_proj_weight"],
            arrays["vllm_self_attn_qkv_proj_weight"],
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            tp=args.tp,
        )

    if "maxtext_mlp_experts_w13_weight" in arrays and "vllm_mlp_experts_w13_weight" in arrays:
        analyze_w13(
            arrays["maxtext_mlp_experts_w13_weight"],
            arrays["vllm_mlp_experts_w13_weight"],
            tp=args.tp,
            num_experts=args.num_experts,
        )

    if "maxtext_mlp_experts_w2_weight" in arrays and "vllm_mlp_experts_w2_weight" in arrays:
        analyze_w2(
            arrays["maxtext_mlp_experts_w2_weight"],
            arrays["vllm_mlp_experts_w2_weight"],
        )


if __name__ == "__main__":
    main()
