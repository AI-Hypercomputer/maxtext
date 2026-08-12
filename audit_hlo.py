import re

with open('/tmp/hlo_dump.hlo', 'r') as f:
    text = f.read()

args_pattern = re.findall(r'(%arg\d+):\s*(tensor<[^>]+>)\s*\{([^}]+)\}', text)
print(f"Total arguments found: {len(args_pattern)}")

pinned_args = []
device_args = []
for name, shape, meta in args_pattern:
    if 'memory_kind = "pinned_host"' in meta:
        pinned_args.append((name, shape))
    elif 'memory_kind = "device"' in meta:
        device_args.append((name, shape))

print(f"Pinned host parameters: {len(pinned_args)}")
print(f"Device tensors (optimizer states, embeddings, inputs): {len(device_args)}")

print("\n--- 1. Pinned Host Parameters (Host Memory Space) ---")
for name, shape in pinned_args[:10]:
    print(f"  {name}: {shape} [pinned_host]")

print("\n--- 2. Device Tensors (HBM Device Memory Space) ---")
for name, shape in device_args[:10]:
    print(f"  {name}: {shape} [device]")

print("\n--- 3. Host-to-Device Placement Annotations / Prefetch ---")
device_placements = re.findall(r'%[a-zA-Z0-9_#:]+\s*=\s*stablehlo\.custom_call\s*@annotate_device_placement\([^)]+\)', text)
print(f"Total @annotate_device_placement calls: {len(device_placements)}")
for dp in device_placements[:8]:
    print(f"  {dp}")

print("\n--- 4. Activation Offload / Host Offloader Warnings in TPU Log ---")
EOF = 1
