r"""Convert weights from FP8/FP4 to BF16 for a DeepSeek HF model.

Example cmd:
python3 deepseek_dequantize.py --input-path <path/to/quantized/ckpt> \
    --output-path <local/path/to/save/new/bf16/ckpt>
"""

import os
import json
import torch
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
from safetensors.torch import load_file, save_file

# Lookup table for E2M1 FP4 (Two e2m1 nibbles packed per int8/uint8 byte)
_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32
)


def weight_dequant_cpu(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given FP8 weight tensor using the provided scale tensor on CPU.
    """
    assert x.dim() == 2 and s.dim() == 2, "Both x and s must be 2D tensors"

    M, N = x.shape

    x = x.to(torch.float32)
    y = torch.empty_like(x, dtype=torch.bfloat16)

    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            row_start = i
            row_end = min(i + block_size, M)
            col_start = j
            col_end = min(j + block_size, N)
            block = x[row_start:row_end, col_start:col_end]
            scale = s[i // block_size, j // block_size]
            y[row_start:row_end, col_start:col_end] = (block * scale).to(torch.bfloat16)

    return y

# reference: https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/integrations/finegrained_fp8.py#L933-L1046
def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpacks packed E2M1 FP4 values (two 4-bit nibbles per byte) into standard float32 values.
    """
    u8 = packed.contiguous().view(torch.uint8)
    low = (u8 & 0xF).long()
    high = ((u8 >> 4) & 0xF).long()

    unpacked = torch.stack([_FP4_E2M1_LUT[low], _FP4_E2M1_LUT[high]], dim=-1)
    return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])

# reference: https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/integrations/finegrained_fp8.py#L933-L1046
def dequantize_mxfp4(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes FP4 (E2M1) or FP8 (E4M3) weights using their block-wise scale grids.
    """
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    is_fp4 = quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype)

    if is_fp4:
        quantized_fp32 = unpack_fp4(quantized)
    else:
        quantized_fp32 = quantized.to(torch.float32)

    rows, cols = quantized_fp32.shape[-2:]
    scale_rows, scale_cols = scales.shape[-2:]

    if rows % scale_rows != 0 or cols % scale_cols != 0:
        raise ValueError(f"Weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols}).")

    block_m = rows // scale_rows
    block_n = cols // scale_cols

    original_shape = quantized_fp32.shape
    q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = scales.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)

    return (q * s).to(torch.bfloat16).reshape(original_shape)


def convert_model(input_path: str, output_path: str, cache_file_num: int = 2):
    """
    Scans, converts, and saves a DeepSeek FP8/FP4 checkpoint directory to BF16.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")

    if not os.path.exists(model_index_file):
        raise FileNotFoundError(f"Could not locate {model_index_file}. Ensure the path is correct.")

    with open(model_index_file, "r", encoding="utf8") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    loaded_files = {}
    converted_scales = []

    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(input_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    print(f"Found {len(safetensor_files)} weight shards to process...")

    for safetensor_file in tqdm(safetensor_files, desc="Converting Shards"):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}

        for name, tensor in current_state_dict.items():
            # Skip scale tensors; they will be integrated directly into the weights
            if name.endswith("_scale_inv") or name.endswith(".scale"):
                continue

            if tensor.dtype in (torch.int8, torch.float8_e4m3fn) or tensor.element_size() == 1:
                # Handle both DeepSeek-V3 (_scale_inv) and V4 (scale) naming conventions
                scale_name_v3 = f"{name}_scale_inv"
                scale_name_v4 = f"{name[:-len('.weight')]}.scale" if name.endswith(".weight") else None

                scale_inv = None
                used_scale_name = None

                try:
                    scale_inv = get_tensor(scale_name_v3)
                    used_scale_name = scale_name_v3
                except KeyError:
                    if scale_name_v4:
                        try:
                            scale_inv = get_tensor(scale_name_v4)
                            used_scale_name = scale_name_v4
                        except KeyError:
                            pass

                if scale_inv is not None:
                    if tensor.dtype == torch.int8:
                        dequantized_tensor = dequantize_mxfp4(tensor, scale_inv)
                    elif tensor.dtype == torch.float8_e4m3fn:
                        dequantized_tensor = weight_dequant_cpu(tensor, scale_inv, block_size=128)

                    new_state_dict[name] = dequantized_tensor
                    converted_scales.append(used_scale_name)
                else:
                    print(f"\nWarning: scale missing for {name}. Keeping original tensor.")
                    new_state_dict[name] = tensor
            else:
                # Keep other non-quantized tensors (like biases, embeds, layer norms) intact in BF16
                new_state_dict[name] = tensor.to(torch.bfloat16)

        save_file(new_state_dict, os.path.join(output_path, file_name))

        # Memory management: keep only the `cache_file_num` most recently used files
        while len(loaded_files) > cache_file_num:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]

    # Clean up JSON Index Map
    print("Saving updated model index map...")
    for scale_name in set(converted_scales):
        if scale_name in weight_map:
            weight_map.pop(scale_name)

    new_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w", encoding="utf8") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

    print(f"Successfully saved dequantized BF16 model to: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Dequantize DeepSeek hybrid checkpoints (FP8/FP4) to BF16.")
    parser.add_argument("--input-path", "--input-fp8-hf-path", type=str, required=True,
                        help="Path to DeepSeek FP8/FP4 Hugging Face folder")
    parser.add_argument("--output-path", "--output-bf16-hf-path", type=str, required=True,
                        help="Directory to save output BF16 weights")
    parser.add_argument("--cache-size", "--cache-file-num", type=int, default=2,
                        help="Max cached files in RAM during indexing lookup")
    args = parser.parse_args()

    convert_model(args.input_path, args.output_path, args.cache_size)

