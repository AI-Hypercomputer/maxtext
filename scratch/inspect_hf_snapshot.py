from huggingface_hub import snapshot_download
import os

model_id = "facebook/DiT-XL-2-256"
token = os.environ.get("HF_TOKEN")

path = snapshot_download(repo_id=model_id, token=token)
print(f"Snapshot path: {path}")

for root, dirs, files in os.walk(path):
    print(f"Root: {root}")
    for f in files:
        print(f"  File: {f}")
