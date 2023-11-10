import jax
from multislice_cloud_tpu_cluster import MultisliceTpuCluster



c = MultisliceTpuCluster()

print(f"{c=}")
print(f"{c.is_env_present()=}")
print(f"{c.get_coordinator_address()=}")
print(f"{c.get_process_count()=}") # requires all workers
print(f"{c.get_process_id()=}") # ???