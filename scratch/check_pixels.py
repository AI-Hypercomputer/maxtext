import numpy as np
from PIL import Image
import sys

def check(img_path):
    try:
        img = Image.open(img_path)
        img_np = np.array(img)
        print(f"Stats for {img_path}:")
        print(f"  Shape: {img_np.shape}")
        print(f"  Dtype: {img_np.dtype}")
        print(f"  Min: {img_np.min()}")
        print(f"  Max: {img_np.max()}")
        print(f"  Mean: {img_np.mean():.2f}")
        print(f"  Std: {img_np.std():.2f}")
        
        # Check unique values
        unique_vals = np.unique(img_np)
        print(f"  Unique values count: {len(unique_vals)}")
        if len(unique_vals) < 10:
            print(f"  Unique values: {unique_vals}")
            
    except Exception as e:
        print(f"Error checking {img_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_pixels.py <image_path>")
    else:
        check(sys.argv[1])
