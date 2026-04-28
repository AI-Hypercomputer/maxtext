import numpy as np
import sys
import os
import glob
import subprocess
import tempfile

def bf16_to_fp32(x):
    # bfloat16 is just the upper 16 bits of float32.
    # So we view as uint16, cast to uint32, shift left by 16, and view as float32.
    return (x.view(np.uint16).astype(np.uint32) << 16).view(np.float32)

def compare_weights(maxtext_path, vllm_path):
    maxtext_w = np.load(maxtext_path)
    vllm_w = np.load(vllm_path)

    if maxtext_w.shape != vllm_w.shape:
        return maxtext_w.shape, vllm_w.shape, "SHAPE MISMATCH", None

    # Handle bfloat16 loaded as void type (V2)
    if maxtext_w.dtype.itemsize == 2:
        print(f"Detected 16-bit floats (dtype={maxtext_w.dtype}). Converting to float32 for comparison.")
        maxtext_w = bf16_to_fp32(maxtext_w)
        vllm_w = bf16_to_fp32(vllm_w)
    else:
        maxtext_w = maxtext_w.astype(np.float32)
        vllm_w = vllm_w.astype(np.float32)

    diff = np.abs(maxtext_w - vllm_w)
    max_diff = np.max(diff)
    
    status = "PASS" if max_diff < 1e-5 else "FAIL"
    
    return maxtext_w.shape, vllm_w.shape, status, max_diff

def process_directory(dir_path):
    print(f"Processing files in {dir_path}...\n")
    
    is_gcs = dir_path.startswith("gs://")
    temp_dir = None
    
    if is_gcs:
        home_dir = os.path.expanduser("~")
        temp_files_parent = os.path.join(home_dir, "workspace", "temp_files")
        os.makedirs(temp_files_parent, exist_ok=True)
        
        temp_dir = tempfile.mkdtemp(dir=temp_files_parent)
        print(f"Created temporary directory for GCS files: {temp_dir}")
        
        # List files using gsutil
        cmd = f"gsutil ls {os.path.join(dir_path, 'maxtext_*.npy')}"
        try:
            output = subprocess.check_output(cmd, shell=True, text=True)
            maxtext_files = output.strip().split('\n')
        except subprocess.CalledProcessError as e:
            print(f"Error listing GCS bucket: {e}")
            return
    else:
        maxtext_files = glob.glob(os.path.join(dir_path, "maxtext_*.npy"))
    
    if not maxtext_files or (len(maxtext_files) == 1 and maxtext_files[0] == ''):
        print("No files starting with 'maxtext_' found.")
        return

    print(f"{'Weight Key':<60} {'MaxText Shape':<20} {'vLLM Shape':<20} {'Status':<10} {'Max Diff':<10}")
    print("-" * 120)

    mismatches = []

    for maxtext_file in sorted(maxtext_files):
        basename = os.path.basename(maxtext_file)
        key = basename.replace("maxtext_", "").replace(".npy", "")
        
        if is_gcs:
            vllm_file_gcs = os.path.join(dir_path, f"vllm_{key}.npy")
            
            # Download files to temp dir
            m_local = os.path.join(temp_dir, basename)
            v_local = os.path.join(temp_dir, f"vllm_{key}.npy")
            
            try:
                subprocess.check_call(f"gsutil cp {maxtext_file} {m_local}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.check_call(f"gsutil cp {vllm_file_gcs} {v_local}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"{key:<60} {'N/A':<20} {'N/A':<20} {'GCS COPY ERROR':<10} {'N/A':<10}")
                continue
                
            maxtext_shape, vllm_shape, status, max_diff = compare_weights(m_local, v_local)
            
            # Clean up local files to save space
            os.remove(m_local)
            os.remove(v_local)
        else:
            vllm_file = os.path.join(dir_path, f"vllm_{key}.npy")
            
            if not os.path.exists(vllm_file):
                print(f"{key:<60} {'N/A':<20} {'N/A':<20} {'MISSING VLLM':<10} {'N/A':<10}")
                continue
                
            maxtext_shape, vllm_shape, status, max_diff = compare_weights(maxtext_file, vllm_file)
        
        shape_str = str(maxtext_shape)
        vllm_shape_str = str(vllm_shape)
        diff_str = f"{max_diff:.6f}" if max_diff is not None else "N/A"
        
        print(f"{key:<60} {shape_str:<20} {vllm_shape_str:<20} {status:<10} {diff_str:<10}")
        
        if status != "PASS":
            if is_gcs:
                # Re-download for analysis if needed
                m_local = os.path.join(temp_dir, basename)
                v_local = os.path.join(temp_dir, f"vllm_{key}.npy")
                subprocess.check_call(f"gsutil cp {maxtext_file} {m_local}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.check_call(f"gsutil cp {vllm_file_gcs} {v_local}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                mismatches.append((key, m_local, v_local, status, max_diff))
            else:
                mismatches.append((key, maxtext_file, vllm_file, status, max_diff))

    print("-" * 120)
    print(f"Total files processed: {len(maxtext_files)}")
    
    if mismatches:
        print("\nDetails for non-passing files:")
        for key, m_file, v_file, status, max_diff in mismatches:
            print(f"\nAnalyzing {key}...")
            maxtext_w = np.load(m_file)
            vllm_w = np.load(v_file)
            
            if maxtext_w.dtype.itemsize == 2:
                maxtext_w = bf16_to_fp32(maxtext_w)
                vllm_w = bf16_to_fp32(vllm_w)
            else:
                maxtext_w = maxtext_w.astype(np.float32)
                vllm_w = vllm_w.astype(np.float32)
                
            if status == "FAIL":
                diff = np.abs(maxtext_w - vllm_w)
                flat_diff = diff.flatten()
                sorted_indices = np.argsort(flat_diff)[::-1]
                
                print("  Top 5 largest differences:")
                for i in range(min(5, len(sorted_indices))):
                    idx = sorted_indices[i]
                    unravel_idx = np.unravel_index(idx, maxtext_w.shape)
                    print(f"    Index {unravel_idx}: MaxText={maxtext_w[unravel_idx]:.6f}, vLLM={vllm_w[unravel_idx]:.6f}, Diff={diff[unravel_idx]:.6f}")
            elif status == "SHAPE MISMATCH":
                print(f"  Shapes: MaxText={maxtext_w.shape}, vLLM={vllm_w.shape}")

    if temp_dir and os.path.exists(temp_dir):
        # Clean up remaining files in temp dir
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        print(f"\nRemoved temporary directory: {temp_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_weights.py <directory_path_or_gs_url>")
    else:
        process_directory(sys.argv[1])
