import torch
import torchvision
from torchvision import transforms
from PIL import Image
import sys

def verify(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(img).unsqueeze(0)

    categories = None
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        categories = weights.meta["categories"]
    except ImportError:
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
    
    model.eval()
    with torch.no_grad():
        out = model(img_t)
    
    prob = torch.nn.functional.softmax(out, dim=1)[0]
    top5_prob, top5_catid = torch.topk(prob, 5)
    
    print(f"Results for {img_path}:")
    for i in range(top5_prob.size(0)):
        idx = top5_catid[i].item()
        name = categories[idx] if categories else "Unknown"
        print(f"  Class ID: {idx}, Label: {name}, Probability: {top5_prob[i].item():.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_image.py <image_path>")
    else:
        verify(sys.argv[1])
