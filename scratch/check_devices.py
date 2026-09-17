import jax
devices = jax.devices()
print("Devices:", devices)
for i, d in enumerate(devices):
    print(f"Device {i}: {d}")
    print(f"  id: {d.id}")
    print(f"  coords: {d.coords}")
    print(f"  core_on_chip: {d.core_on_chip}")

