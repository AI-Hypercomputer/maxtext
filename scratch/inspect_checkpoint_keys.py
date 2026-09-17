import orbax.checkpoint as ocp
import sys
import os

import orbax.checkpoint as ocp
import sys
from etils import epath

def inspect(checkpoint_path):
    print(f"Inspecting checkpoint at: {checkpoint_path}")
    path = epath.Path(checkpoint_path)
    
    # Try to list files/directories directly using epath
    if path.exists():
        print("Found checkpoint path.")
        params_dir = path / "params"
        if params_dir.exists():
            print(f"Found params dir: {params_dir}")
            for p in params_dir.iterdir():
                # Print only first few or filtered
                if "vae_decoder" in p.name:
                   print(p.name)
        else:
            print("No 'params' subdirectory found directly under path.")
            for p in path.iterdir():
                 if "vae" in p.name:
                     print(p.name)
    else:
        print("Checkpoint path does not exist!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint_keys.py <checkpoint_path>")
    else:
        inspect(sys.argv[1])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint_keys.py <checkpoint_path>")
    else:
        inspect(sys.argv[1])
