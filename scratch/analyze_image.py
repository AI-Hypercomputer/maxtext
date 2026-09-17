import numpy as np
from PIL import Image
import sys

def analyze(img_path):
    img = Image.open(img_path)
    img_np = np.array(img)
    print(f"Stats for {img_path}:")
    print(f"  Shape: {img_np.shape}")
    print(f"  Dtype: {img_np.dtype}")
    print(f"  Min: {img_np.min()}")
    print(f"  Max: {img_np.max()}")
    print(f"  Mean: {img_np.mean():.2f}")
    print(f"  Std: {img_np.std():.2f}")
    
    # Check if it looks like noise (e.g. high entropy, uniform distribution)
    # Print a small 4x4 block from center
    h, w, c = img_np.shape
    print("  Center 4x4 block (R channel):")
    print(img_np[h//2-2:h//2+2, w//2-2:w//2+2, 0])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_image.py <image_path>")
    else:
        analyze(sys.argv[1])
