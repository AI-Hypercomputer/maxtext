import sys
import numpy as np
from PIL import Image

def analyze_image(path):
    try:
        img = Image.open(path)
        img_np = np.array(img)
        print(f"Image: {path}")
        print(f"  Shape: {img_np.shape}")
        print(f"  Stats: min={img_np.min()}, max={img_np.max()}, mean={img_np.mean():.2f}, std={img_np.std():.2f}")
        
        # Check if image is completely solid color
        if img_np.std() < 1.0:
            print("  WARNING: Image is likely a solid color (low variance).")
        
        # Check if image is likely pure noise (very high spatial frequency without structure)
        # Simple heuristic: difference between adjacent pixels
        diff_h = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :])
        diff_v = np.abs(img_np[1:, :, :] - img_np[:-1, :, :])
        print(f"  Spatial diff mean: h={diff_h.mean():.2f}, v={diff_v.mean():.2f}")
        
        if diff_h.mean() > 50 and diff_v.mean() > 50:
             print("  WARNING: Image might be high frequency noise.")
             
        print("-" * 20)
    except Exception as e:
        print(f"Error analyzing {path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_images.py <image_path1> <image_path2> ...")
    else:
        for path in sys.argv[1:]:
            analyze_image(path)
