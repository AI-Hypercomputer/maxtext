import jax;

print("Calling JDI...", flush=True)
jax.distributed.initialize()
print("JDI successful!!", flush=True)