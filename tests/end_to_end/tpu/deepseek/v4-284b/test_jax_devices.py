import os
# Configure XLA flags before importing JAX to simulate 16 CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

print("Importing JAX...")
import jax
jax.config.update("jax_platforms", "cpu")
print("Default backend:", jax.default_backend())
print("Number of devices:", jax.device_count())
print("Devices:", jax.devices())
print("Successfully initialized all 16 JAX simulated CPU devices!")
