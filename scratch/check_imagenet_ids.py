import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

def check():
    weights = ResNet18_Weights.DEFAULT
    categories = weights.meta["categories"]
    
    print(f"Index 2: {categories[2]}")
    print(f"Index 879: {categories[879]}")
    
    # Also find indices for "white shark" and "umbrella"
    for i, cat in enumerate(categories):
        if "shark" in cat:
            print(f"Found shark at index {i}: {cat}")
        if "umbrella" in cat:
            print(f"Found umbrella at index {i}: {cat}")

if __name__ == "__main__":
    check()
